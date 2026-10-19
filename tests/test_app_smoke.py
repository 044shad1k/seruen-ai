"""The Streamlit app must survive every preset scenario without an exception.

Runs the real script through Streamlit's AppTest with weather turned off, so
the test stays offline and never touches the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "seruen" / "app" / "streamlit_app.py"
DATASET = Path(__file__).resolve().parents[1] / "data" / "pois.json"

pytestmark = pytest.mark.skipif(
    not DATASET.exists(), reason="dataset not collected yet (run: make data)"
)

WEATHER_CHECKBOX = "Учитывать погоду"


def fresh_app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    for checkbox in app.checkbox:
        if checkbox.label == WEATHER_CHECKBOX:
            checkbox.set_value(False)
    app.run()
    return app


def test_app_starts_without_a_query():
    app = fresh_app()
    assert not app.exception
    assert app.title[0].value.startswith("🚶")
    # the five brief scenarios are one click away
    assert len(app.button) == 5


@pytest.mark.parametrize("index", range(5))
def test_every_preset_scenario_answers(index: int):
    app = fresh_app()
    app.button[index].click().run()
    assert not app.exception, [e.value for e in app.exception]

    texts = [block.value for block in app.markdown]
    assert any("Почему:" in text for text in texts), "ни одной карточки с объяснением"
    assert any("Что делать:" in text for text in texts), "нет блока «что делать»"
    # the data-used panel always reports how many candidates were considered
    labels = [metric.label for metric in app.metric]
    assert "Кандидатов после фильтров" in labels
    assert "Ушло в модель" in labels


def test_answer_never_claims_open_without_data():
    app = fresh_app()
    app.button[0].click().run()
    for text in [caption.value for caption in app.caption]:
        if "часы работы не подтверждены" in text or "нет часов работы" in text:
            assert "🟢 открыто" not in text
