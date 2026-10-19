"""A deliberately small subset of the OSM `opening_hours` syntax.

Design rule from the brief: never claim "open now" without data. Anything this
parser does not fully understand becomes `unknown`, never `open`.

Supported:
    24/7
    Mo-Fr 09:00-18:00
    Mo-Fr 09:00-18:00; Sa 10:00-16:00
    Mo-Su 10:00-22:00
    Mo-Su 20:00-02:00            (crosses midnight)
    Mo-Fr 09:00-12:00,13:00-18:00
    Mo,We,Fr 10:00-19:00
    Mo-Fr 09:00-18:00; Su off
    PH/SH rules are ignored (holidays are out of scope, not a parse failure)
Everything else -> unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

OPEN = "open"
CLOSED = "closed"
UNKNOWN = "unknown"

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_INDEX = {day.lower(): i for i, day in enumerate(DAYS)}
HOLIDAY_SELECTORS = {"ph", "sh", "ph off", "sh off"}

_TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$")
_DAY_PART_RE = re.compile(r"^[A-Za-z]{2}(\s*-\s*[A-Za-z]{2})?$")

STATUS_LABEL_RU = {
    OPEN: "открыто",
    CLOSED: "закрыто",
    UNKNOWN: "часы работы неизвестны",
}


@dataclass
class HoursResult:
    """Opening status plus the confidence we are allowed to show."""

    status: str  # open | closed | unknown
    confidence: str  # high | medium | low
    raw: str | None
    detail: str
    minutes_until_close: int | None = None

    @property
    def label_ru(self) -> str:
        return STATUS_LABEL_RU[self.status]

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "raw": self.raw,
            "detail": self.detail,
            "minutes_until_close": self.minutes_until_close,
        }


def _parse_days(part: str) -> list[int] | None:
    """`Mo-Fr` / `Mo,We` / `Sa` -> weekday indices (Mo=0). None if unparsable."""
    result: list[int] = []
    for chunk in part.split(","):
        chunk = chunk.strip()
        if not chunk or not _DAY_PART_RE.match(chunk):
            return None
        if "-" in chunk:
            start_raw, end_raw = (piece.strip().lower() for piece in chunk.split("-", 1))
            if start_raw not in DAY_INDEX or end_raw not in DAY_INDEX:
                return None
            start, end = DAY_INDEX[start_raw], DAY_INDEX[end_raw]
            if start <= end:
                result.extend(range(start, end + 1))
            else:  # wraps the week, e.g. Sa-Mo
                result.extend(list(range(start, 7)) + list(range(0, end + 1)))
        else:
            if chunk.lower() not in DAY_INDEX:
                return None
            result.append(DAY_INDEX[chunk.lower()])
    return sorted(set(result))


def _parse_time_ranges(part: str) -> list[tuple[int, int]] | None:
    """`09:00-18:00,19:00-22:00` -> [(540, 1080), (1140, 1320)] in minutes."""
    ranges: list[tuple[int, int]] = []
    for chunk in part.split(","):
        match = _TIME_RANGE_RE.match(chunk.strip())
        if not match:
            return None
        h1, m1, h2, m2 = (int(group) for group in match.groups())
        if not (0 <= h1 <= 24 and 0 <= h2 <= 24 and m1 < 60 and m2 < 60):
            return None
        ranges.append((h1 * 60 + m1, h2 * 60 + m2))
    return ranges


def _split_rule(rule: str) -> tuple[list[int], list[tuple[int, int]] | str] | None:
    """One `;`-separated rule -> (weekdays, time ranges) or (weekdays, "off")."""
    rule = rule.strip()
    if not rule:
        return None
    lowered = rule.lower()
    if lowered in {"24/7", "24x7", "00:00-24:00"}:
        return list(range(7)), [(0, 24 * 60)]

    tokens = rule.split()
    if not tokens:
        return None

    # a rule may start with a day selector; without one it applies to every day
    days = _parse_days(tokens[0])
    rest = tokens[1:] if days is not None else tokens
    if days is None:
        days = list(range(7))

    if not rest:
        return None
    rest_joined = " ".join(rest).strip().lower()
    if rest_joined in {"off", "closed"}:
        return days, "off"
    if rest_joined in {"24/7", "open", "00:00-24:00"}:
        return days, [(0, 24 * 60)]

    ranges = _parse_time_ranges(" ".join(rest).replace(" ", ""))
    if ranges is None:
        return None
    return days, ranges


def hours_status(raw: str | None, when: datetime) -> HoursResult:
    """Opening status of a POI at `when`, from its raw `opening_hours` tag."""
    if raw is None or not str(raw).strip():
        return HoursResult(
            status=UNKNOWN,
            confidence="low",
            raw=None,
            detail="в данных OSM нет часов работы",
        )

    raw_text = str(raw).strip()
    weekday = when.weekday()  # Mo = 0
    minute_of_day = when.hour * 60 + when.minute

    open_ranges_today: list[tuple[int, int]] = []
    explicit_off_today = False
    saw_any_rule = False

    for rule in raw_text.split(";"):
        rule = rule.strip()
        if not rule:
            continue
        if rule.lower() in HOLIDAY_SELECTORS or rule.lower().startswith(("ph ", "sh ")):
            continue  # holidays: out of scope, but not a parse failure
        parsed = _split_rule(rule)
        if parsed is None:
            return HoursResult(
                status=UNKNOWN,
                confidence="low",
                raw=raw_text,
                detail=f"формат часов не распознан: {raw_text}",
            )
        saw_any_rule = True
        days, spec = parsed
        if weekday not in days:
            continue
        if spec == "off":
            explicit_off_today = True
            open_ranges_today = []
            continue
        explicit_off_today = False
        open_ranges_today.extend(spec)

    if not saw_any_rule:
        return HoursResult(
            status=UNKNOWN,
            confidence="low",
            raw=raw_text,
            detail=f"формат часов не распознан: {raw_text}",
        )

    # midnight-crossing ranges from yesterday can still be running now
    yesterday = (weekday - 1) % 7
    carry_over: list[tuple[int, int]] = []
    for rule in raw_text.split(";"):
        parsed = _split_rule(rule)
        if parsed is None:
            continue
        days, spec = parsed
        if spec == "off" or yesterday not in days:
            continue
        for start, end in spec:
            if end < start:  # e.g. 20:00-02:00 -> runs into today
                carry_over.append((0, end))

    for start, end in open_ranges_today:
        if end < start:  # crosses midnight: open until 24:00 today
            if minute_of_day >= start:
                return HoursResult(
                    OPEN,
                    "high",
                    raw_text,
                    f"по данным OSM: {raw_text}",
                    minutes_until_close=(24 * 60 - minute_of_day) + end,
                )
        elif start <= minute_of_day < end:
            return HoursResult(
                OPEN,
                "high",
                raw_text,
                f"по данным OSM: {raw_text}",
                minutes_until_close=end - minute_of_day,
            )

    for start, end in carry_over:
        if start <= minute_of_day < end:
            return HoursResult(
                OPEN,
                "high",
                raw_text,
                f"по данным OSM: {raw_text}",
                minutes_until_close=end - minute_of_day,
            )

    detail = f"по данным OSM: {raw_text}"
    if explicit_off_today:
        detail = f"сегодня выходной по данным OSM: {raw_text}"
    return HoursResult(CLOSED, "high", raw_text, detail)
