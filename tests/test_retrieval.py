"""Retrieval hard filters: closed is excluded, unknown is kept and flagged."""

from __future__ import annotations

from datetime import datetime

from seruen.core.hours import OPEN, UNKNOWN
from seruen.core.intent import parse_intent_regex
from seruen.core.retrieval import diversify, retrieve

MONDAY_12 = datetime(2026, 10, 19, 12, 0)
ORIGIN = (51.1283, 71.4305)  # Baiterek


def poi(
    poi_id: str,
    name: str,
    category: str,
    lat: float,
    lon: float,
    hours_raw: str | None = None,
    tags: list[str] | None = None,
    wheelchair: str = "unknown",
    indoor: bool | None = None,
    description: str = "",
) -> dict:
    return {
        "id": poi_id,
        "name": {"ru": name, "kk": name, "en": name},
        "lat": lat,
        "lon": lon,
        "category": category,
        "osm_subtype": category,
        "tags": tags or [],
        "description": description or f"{category} в Астане.",
        "description_source": "osm_tags",
        "hours_raw": hours_raw,
        "wheelchair": wheelchair,
        "indoor": indoor,
        "website": None,
        "phone": None,
        "wikidata": None,
        "source": {"url": "https://example.invalid", "license": "ODbL", "fetched_at": None},
        "evidence": [],
    }


# ~400 m north of the origin, well inside any radius
NEAR = (51.1320, 71.4305)
FAR = (51.2000, 71.4305)  # ~8 km away

DATASET = (
    poi("node/1", "Открытый парк", "park", *NEAR, hours_raw="Mo-Su 08:00-22:00"),
    poi("node/2", "Закрытый музей", "museum", *NEAR, hours_raw="Mo-Su 19:00-22:00"),
    poi("node/3", "Парк без часов", "park", *NEAR, hours_raw=None),
    poi("node/4", "Далёкий парк", "park", *FAR, hours_raw="24/7"),
    poi("node/5", "Кафе с ступенями", "cafe", *NEAR, hours_raw="24/7", wheelchair="no"),
    poi("node/6", "Доступное кафе", "cafe", *NEAR, hours_raw="24/7", wheelchair="yes"),
)


def ids(result) -> list[str]:
    return [candidate.poi["id"] for candidate in result.candidates]


def test_closed_poi_is_excluded():
    intent = parse_intent_regex("хочу в парк или музей на 60 минут")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    assert "node/2" not in ids(result)
    assert result.stats["excluded_closed"] >= 1


def test_unknown_hours_are_kept_and_flagged():
    intent = parse_intent_regex("хочу в парк на 60 минут")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    assert "node/3" in ids(result)
    flagged = [c for c in result.candidates if c.poi["id"] == "node/3"][0]
    assert flagged.hours.status == UNKNOWN
    assert "часы работы не подтверждены" in flagged.uncertainties()
    assert result.unknown_hours_count >= 1


def test_distance_filter_drops_far_poi():
    intent = parse_intent_regex("хочу в парк на 45 минут")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    assert "node/4" not in ids(result)
    assert result.stats["excluded_too_far"] >= 1


def test_open_poi_outranks_unknown_hours_all_else_equal():
    intent = parse_intent_regex("хочу в парк на 60 минут")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    scores = {c.poi["id"]: c.score for c in result.candidates}
    assert scores["node/1"] > scores["node/3"]


def test_wheelchair_no_is_excluded_when_required():
    intent = parse_intent_regex("нужно кафе без ступеней, у меня коляска, 60 минут")
    assert intent.needs_wheelchair
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    assert "node/5" not in ids(result)
    assert "node/6" in ids(result)


def test_avoided_category_is_excluded():
    intent = parse_intent_regex("хочу погулять 60 минут, без кафе")
    assert "cafe" in intent.categories_avoided
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    assert "node/6" not in ids(result)


def test_candidate_card_hides_user_location_and_keeps_hours_status():
    intent = parse_intent_regex("парк на час")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    card = result.candidates[0].card()
    assert set(card) == {
        "poi_id", "name", "category", "description", "tags", "hours_status",
        "hours_raw", "distance_m", "walk_min", "indoor", "wheelchair",
    }
    assert card["hours_status"] in {OPEN, UNKNOWN}
    assert "lat" not in card and "lon" not in card


def test_diversify_limits_one_category():
    intent = parse_intent_regex("погулять час")
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=DATASET)
    chosen = diversify(result.candidates, limit=3, max_per_category=1)
    categories = [c.poi["category"] for c in chosen]
    assert len(categories) == len(set(categories))


def test_empty_result_when_nothing_in_range():
    intent = parse_intent_regex("музей на 20 минут")
    far_only = (poi("node/9", "Очень далеко", "museum", *FAR, hours_raw="24/7"),)
    result = retrieve(intent, *ORIGIN, MONDAY_12, pois=far_only)
    assert result.candidates == []
