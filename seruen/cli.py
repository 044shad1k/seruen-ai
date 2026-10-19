"""Seruen AI command line.

    python -m seruen.cli --preset mega --query "хочу погулять 45 минут" --demo
    python -m seruen.cli --lat 51.1283 --lon 71.4305 --query "тихое кафе с розетками"

`advise()` is the whole pipeline and is also what the Streamlit app calls.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from seruen.core import retrieval
from seruen.core import weather as weather_mod
from seruen.core.generate import Answer, generate
from seruen.core.intent import Intent, parse_intent
from seruen.core.retrieval import RetrievalResult
from seruen.core.route import Route, build_route, describe_route

# Starting points for demos. Coordinates checked against the collected OSM data
# (Mega Silk Way, Baiterek, Khan Shatyr came out of data/pois.json).
PRESETS: dict[str, tuple[str, float, float]] = {
    "mega": ("Mega Silk Way", 51.0891, 71.4073),
    "expo": ("EXPO / Нурлы Жол", 51.0893, 71.4178),
    "baiterek": ("Байтерек", 51.1283, 71.4305),
    "khan": ("Хан Шатыр", 51.1325, 71.4039),
    "airport": ("Аэропорт Нурсултан Назарбаев", 51.0222, 71.4669),
    "nu": ("Nazarbayev University", 51.0905, 71.3978),
}

DEFAULT_ORIGIN = PRESETS["baiterek"]


@dataclass
class Advice:
    """Everything one request produced — answer, evidence and the data trail."""

    intent: Intent
    answer: Answer
    retrieval: RetrievalResult
    route: Route | None
    weather: dict | None
    origin: tuple[float, float]
    origin_label: str
    when: datetime

    def data_used(self) -> dict:
        """The "Какие данные использованы" panel. Location is rounded on purpose."""
        from seruen.core.geo import round_coords

        lat, lon = round_coords(*self.origin)
        return {
            "location_rounded": [lat, lon],
            "location_label": self.origin_label,
            "local_time": self.when.strftime("%Y-%m-%d %H:%M"),
            "time_budget_min": self.intent.time_budget_min,
            "filters": self.intent.filters_ru(),
            "dataset_size": self.retrieval.stats["dataset_size"],
            "search_radius_m": self.retrieval.stats["radius_m"],
            "candidates_after_filters": self.retrieval.stats["passed_filters"],
            "candidates_shown_to_model": self.retrieval.stats.get("returned", 0),
            "excluded_closed": self.retrieval.stats["excluded_closed"],
            "excluded_too_far": self.retrieval.stats["excluded_too_far"],
            "excluded_no_time": self.retrieval.stats["excluded_no_time"],
            "unknown_hours_kept": self.retrieval.stats["unknown_hours_kept"],
            "radius_relaxed": self.retrieval.stats["relaxed"],
            "weather": weather_mod.describe(self.weather),
            "intent_source": self.intent.source,
            "answer_mode": self.answer.mode,
            "model": self.answer.model or "—",
            "dataset_license": "OpenStreetMap, ODbL — © OpenStreetMap contributors",
        }

    def as_dict(self) -> dict:
        return {
            "query": self.intent.raw_query,
            "intent": self.intent.as_dict(),
            "answer": self.answer.as_dict(),
            "route": self.route.as_dict() if self.route else None,
            "data_used": self.data_used(),
        }


def advise(
    query: str,
    lat: float,
    lon: float,
    budget_min: int | None = None,
    when: datetime | None = None,
    demo: bool | None = None,
    use_weather: bool = True,
    origin_label: str = "указанная точка",
    dataset_path: str | Path = retrieval.DEFAULT_DATASET,
    max_picks: int = 3,
    language: str | None = None,
    overrides: dict | None = None,
) -> Advice:
    """intent -> weather -> retrieval -> generation -> route.

    `language` forces the answer language (the UI switch); `overrides` lets the
    UI set intent fields the user ticked by hand, e.g. {"with_child": True}.
    """
    when = when or datetime.now()
    if demo is None:
        demo = os.environ.get("SERUEN_DEMO", "0") == "1" or not os.environ.get("GOOGLE_API_KEY")

    intent = parse_intent(query, budget_min=budget_min, use_llm=not demo)
    if language in {"ru", "kk", "en"}:
        intent.language = language
    for field_name, value in (overrides or {}).items():
        if hasattr(intent, field_name) and value not in (None, False, 0, ""):
            setattr(intent, field_name, value)
    current_weather = weather_mod.get_weather(lat, lon) if use_weather else None

    result = retrieval.retrieve(
        intent, lat, lon, when, weather=current_weather, dataset_path=dataset_path
    )
    answer = generate(
        intent,
        result.candidates,
        when,
        radius_m=result.radius_m,
        weather=current_weather,
        demo=demo,
    )
    route = build_route(
        answer.picks[:max_picks],
        lat,
        lon,
        return_to_start=intent.return_to_start,
        budget_min=intent.time_budget_min,
    )

    return Advice(
        intent=intent,
        answer=answer,
        retrieval=result,
        route=route,
        weather=current_weather,
        origin=(lat, lon),
        origin_label=origin_label,
        when=when,
    )


# --- printing --------------------------------------------------------------

SOURCE_LABEL = {
    "poi_data": "источник: данные POI",
    "generated_idea": "идея сгенерирована",
}
CONFIDENCE_LABEL = {"high": "высокая", "medium": "средняя", "low": "низкая"}
HOURS_LABEL = {"open": "открыто", "closed": "закрыто", "unknown": "неизвестно"}


def format_advice(advice: Advice) -> str:
    lines: list[str] = []
    lines.append(f"Запрос: {advice.intent.raw_query}")
    lines.append(f"Старт: {advice.origin_label} · {advice.when.strftime('%H:%M')} · бюджет {advice.intent.time_budget_min} мин")
    lines.append("")

    if not advice.answer.picks:
        lines.append(advice.answer.note)
        return "\n".join(lines)

    for number, pick in enumerate(advice.answer.picks, 1):
        lines.append(f"{number}. {pick.name}  —  {pick.distance_m} м, {pick.walk_min} мин пешком")
        lines.append(f"   Почему: {pick.why} [{SOURCE_LABEL.get(pick.source, pick.source)}]")
        lines.append(f"   Что делать: {pick.activity} [идея]")
        lines.append(
            f"   На месте: ~{pick.visit_minutes} мин · часы: {HOURS_LABEL.get(pick.hours.get('status'), '?')}"
            f" ({pick.hours.get('detail')}) · уверенность: {CONFIDENCE_LABEL.get(pick.confidence, pick.confidence)}"
        )
        if pick.uncertainties:
            lines.append(f"   Не проверено: {', '.join(pick.uncertainties)}")
        lines.append("")

    if advice.route:
        lines.append(describe_route(advice.route, advice.intent.language))
        lines.append("")

    lines.append(f"Примечание: {advice.answer.note}")
    if advice.answer.fallback_reason:
        lines.append(f"Режим ответа: {advice.answer.mode} ({advice.answer.fallback_reason})")
    else:
        lines.append(f"Режим ответа: {advice.answer.mode}")
    lines.append("")

    used = advice.data_used()
    lines.append("Какие данные использованы:")
    lines.append(f"  местоположение (округлено): {used['location_rounded'][0]}, {used['location_rounded'][1]} ({used['location_label']})")
    lines.append(f"  время: {used['local_time']} · фильтры: {', '.join(used['filters'])}")
    lines.append(f"  датасет: {used['dataset_size']} POI, OSM/ODbL · радиус поиска: {used['search_radius_m']} м"
                 + (" (расширен автоматически)" if used["radius_relaxed"] else ""))
    lines.append(f"  прошли фильтры: {used['candidates_after_filters']}, показано модели: {used['candidates_shown_to_model']}")
    lines.append(f"  отброшено: закрыто {used['excluded_closed']}, далеко {used['excluded_too_far']}, не влезает по времени {used['excluded_no_time']}")
    lines.append(f"  часы неизвестны у кандидатов: {used['unknown_hours_kept']} (помечены, не выдаются за «открыто»)")
    lines.append(f"  погода: {used['weather']}")
    lines.append(f"  разбор запроса: {used['intent_source']} · генерация: {used['answer_mode']} · модель: {used['model']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seruen AI — ИИ-помощник по прогулкам в Астане",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пресеты точек старта: " + ", ".join(PRESETS),
    )
    parser.add_argument("query", nargs="?", help="запрос пользователя")
    parser.add_argument("--query", dest="query_opt", help="запрос пользователя (альтернатива)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="готовая точка старта")
    parser.add_argument("--budget", type=int, help="бюджет времени в минутах")
    parser.add_argument("--demo", action="store_true", help="только шаблоны, без обращения к модели")
    parser.add_argument("--now", help="переопределить время, например '2026-10-19 23:30'")
    parser.add_argument("--no-weather", action="store_true", help="не запрашивать погоду")
    parser.add_argument("--json", action="store_true", help="вывести JSON вместо карточек")
    parser.add_argument("--dataset", default=str(retrieval.DEFAULT_DATASET))
    args = parser.parse_args(argv)

    query = args.query_opt or args.query
    if not query:
        parser.error("нужен запрос: python -m seruen.cli \"Я у Mega Silk Way, хочу погулять 45 минут\"")

    if args.preset:
        label, lat, lon = PRESETS[args.preset]
    elif args.lat is not None and args.lon is not None:
        label, lat, lon = "указанная точка", args.lat, args.lon
    else:
        label, lat, lon = DEFAULT_ORIGIN
        label += " (точка по умолчанию)"

    when = datetime.now()
    if args.now:
        for fmt in ("%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M", "%H:%M"):
            try:
                parsed = datetime.strptime(args.now, fmt)
                when = (
                    datetime.now().replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
                    if fmt == "%H:%M"
                    else parsed
                )
                break
            except ValueError:
                continue
        else:
            parser.error(f"не понял время: {args.now}")

    advice = advise(
        query,
        lat,
        lon,
        budget_min=args.budget,
        when=when,
        demo=True if args.demo else None,
        use_weather=not args.no_weather,
        origin_label=label,
        dataset_path=args.dataset,
    )

    if args.json:
        print(json.dumps(advice.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_advice(advice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
