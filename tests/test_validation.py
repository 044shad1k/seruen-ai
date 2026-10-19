"""Generation guardrails: a model may only point at candidates we retrieved."""

from __future__ import annotations

from datetime import datetime

from seruen.core.generate import (
    MODE_TEMPLATE,
    SOURCE_IDEA,
    SOURCE_POI,
    generate,
    template_answer,
    validate_picks,
)
from seruen.core.hours import hours_status
from seruen.core.intent import parse_intent_regex
from seruen.core.retrieval import Candidate
from tests.test_retrieval import poi

MONDAY_12 = datetime(2026, 10, 19, 12, 0)


def make_candidate(poi_id: str, name: str, category: str, hours_raw: str | None) -> Candidate:
    record = poi(poi_id, name, category, 51.13, 71.43, hours_raw=hours_raw)
    return Candidate(
        poi=record,
        distance_m=400.0,
        walk_min=7.4,
        hours=hours_status(hours_raw, MONDAY_12),
    )


CANDIDATES = [
    make_candidate("node/1", "Парк", "park", "Mo-Su 08:00-22:00"),
    make_candidate("node/2", "Кафе", "cafe", None),
]
INTENT = parse_intent_regex("хочу погулять 60 минут")


def test_unknown_poi_id_is_dropped():
    picks, dropped = validate_picks(
        [
            {"poi_id": "node/999", "why": "выдуманное место", "activity": "x",
             "visit_minutes": 20, "confidence": "high", "source": SOURCE_POI},
            {"poi_id": "node/1", "why": "рядом и открыто", "activity": "пройтись",
             "visit_minutes": 20, "confidence": "high", "source": SOURCE_POI},
        ],
        CANDIDATES,
        INTENT,
    )
    assert [p.poi_id for p in picks] == ["node/1"]
    assert dropped == 1


def test_duplicate_ids_are_dropped():
    raw = {"poi_id": "node/1", "why": "ок", "activity": "a", "visit_minutes": 20,
           "confidence": "medium", "source": SOURCE_POI}
    picks, dropped = validate_picks([raw, dict(raw)], CANDIDATES, INTENT)
    assert len(picks) == 1
    assert dropped == 1


def test_all_invalid_ids_leaves_nothing():
    picks, dropped = validate_picks(
        [{"poi_id": "way/404", "why": "нет такого", "activity": "a", "visit_minutes": 10,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks == []
    assert dropped == 1


def test_confidence_cannot_exceed_data_quality():
    # node/2 has no opening hours, so "high" must be downgraded
    picks, _ = validate_picks(
        [{"poi_id": "node/2", "why": "тихое место", "activity": "кофе", "visit_minutes": 25,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks[0].confidence == "medium"
    assert picks[0].hours["status"] == "unknown"


def test_bogus_enum_values_are_repaired():
    picks, _ = validate_picks(
        [{"poi_id": "node/1", "why": "ок", "activity": "a", "visit_minutes": 9999,
          "confidence": "очень высокая", "source": "выдумка"}],
        CANDIDATES,
        INTENT,
    )
    assert picks[0].confidence == "medium"
    assert picks[0].source == SOURCE_IDEA
    assert picks[0].visit_minutes <= INTENT.time_budget_min


def test_pick_without_why_is_dropped():
    picks, dropped = validate_picks(
        [{"poi_id": "node/1", "why": "   ", "activity": "a", "visit_minutes": 20,
          "confidence": "high", "source": SOURCE_POI}],
        CANDIDATES,
        INTENT,
    )
    assert picks == []
    assert dropped == 1


def test_no_more_than_three_picks():
    candidates = [make_candidate(f"node/{i}", f"Место {i}", "park", "24/7") for i in range(6)]
    raw = [
        {"poi_id": f"node/{i}", "why": "ок", "activity": "a", "visit_minutes": 15,
         "confidence": "high", "source": SOURCE_POI}
        for i in range(6)
    ]
    picks, _ = validate_picks(raw, candidates, INTENT)
    assert len(picks) == 3


def test_template_answer_is_labelled_and_sourced():
    answer = template_answer(INTENT, CANDIDATES, radius_m=800)
    assert answer.mode == MODE_TEMPLATE
    assert 1 <= len(answer.picks) <= 3
    for pick in answer.picks:
        assert pick.source == SOURCE_POI
        assert pick.activity_source == SOURCE_IDEA
        assert pick.why and pick.activity
        assert pick.hours["status"] in {"open", "unknown"}


def test_demo_mode_never_calls_the_model():
    answer = generate(INTENT, CANDIDATES, MONDAY_12, radius_m=800, demo=True)
    assert answer.mode == MODE_TEMPLATE
    assert answer.model is None
    assert answer.fallback_reason == "demo-режим"


def test_no_candidates_gives_an_honest_empty_answer():
    answer = generate(INTENT, [], MONDAY_12, radius_m=800, demo=True)
    assert answer.picks == []
    assert "не нашлось" in answer.note
