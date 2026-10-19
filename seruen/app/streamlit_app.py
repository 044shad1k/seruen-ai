"""Seruen AI — Streamlit interface: chat, map, preset scenarios, data panel.

    streamlit run seruen/app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pydeck as pdk
import streamlit as st

# allow `streamlit run seruen/app/streamlit_app.py` from the repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seruen.cli import PRESETS, advise  # noqa: E402
from seruen.core import weather as weather_mod  # noqa: E402
from seruen.core.retrieval import DEFAULT_DATASET  # noqa: E402
from seruen.core.route import describe_route  # noqa: E402

DATASET = ROOT / DEFAULT_DATASET

SOURCE_LABEL = {
    "poi_data": "источник: данные POI",
    "generated_idea": "идея сгенерирована",
}
CONFIDENCE_BADGE = {"high": "🟢 высокая", "medium": "🟡 средняя", "low": "🟠 низкая"}
HOURS_BADGE = {
    "open": "🟢 открыто",
    "closed": "🔴 закрыто",
    "unknown": "⚪ часы неизвестны",
}
CATEGORY_RU = {
    "park": "парк",
    "cafe": "кафе",
    "restaurant": "ресторан",
    "museum": "музей",
    "landmark": "достопримечательность",
    "viewpoint": "смотровая",
    "playground": "детская площадка",
    "mall": "ТЦ",
    "theatre": "театр/кино",
    "library": "библиотека",
    "sport": "спорт",
    "other": "место",
}

# the five scenarios from the brief, one click each
SCENARIOS = [
    {
        "label": "🛍 У Mega Silk Way, 45 минут",
        "query": "Я у Mega Silk Way, хочу погулять 45 минут",
        "preset": "mega",
        "budget": 45,
    },
    {
        "label": "💻 Тихое кафе с розетками",
        "query": "ищу тихое кафе с розетками поработать за ноутбуком, час",
        "preset": "baiterek",
        "budget": 60,
    },
    {
        "label": "👶 С ребёнком и коляской",
        "query": "гуляю с ребёнком и коляской, есть час, куда пойти",
        "preset": "khan",
        "budget": 60,
    },
    {
        "label": "✈️ 2 часа до самолёта",
        "query": "я турист, до самолёта 2 часа, хочу посмотреть что-то рядом",
        "preset": "airport",
        "budget": 120,
    },
    {
        "label": "🌙 Необычное ночью",
        "query": "хочу чего-то необычного ночью, 90 минут",
        "preset": "baiterek",
        "budget": 90,
        "time": "23:30",
    },
]


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("preset", "baiterek")
    st.session_state.setdefault("budget", 60)
    st.session_state.setdefault("clock", None)


def apply_scenario(scenario: dict) -> None:
    st.session_state["preset"] = scenario["preset"]
    st.session_state["budget"] = scenario["budget"]
    st.session_state["clock"] = scenario.get("time")
    st.session_state["pending"] = scenario["query"]


def resolve_when(clock: str | None) -> datetime:
    now = datetime.now()
    if not clock:
        return now
    try:
        hour, minute = (int(part) for part in clock.split(":"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError:
        return now


def render_map(advice) -> None:
    """User position, the picks, everything else retrieval considered, route."""
    lat, lon = advice.origin
    picks = advice.answer.picks

    considered = [
        {
            "lat": candidate.poi["lat"],
            "lon": candidate.poi["lon"],
            "name": candidate.name(),
            "kind": "кандидат",
        }
        for candidate in advice.retrieval.candidates
        if candidate.poi["id"] not in {p.poi_id for p in picks}
    ]
    chosen = [
        {
            "lat": pick.lat,
            "lon": pick.lon,
            "name": f"{number}. {pick.name}",
            "kind": "рекомендация",
        }
        for number, pick in enumerate(picks, 1)
    ]
    me = [{"lat": lat, "lon": lon, "name": advice.origin_label, "kind": "вы"}]

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=considered,
            get_position="[lon, lat]",
            get_fill_color=[150, 150, 150, 110],
            get_radius=55,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=chosen,
            get_position="[lon, lat]",
            get_fill_color=[214, 69, 65, 230],
            get_radius=110,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=me,
            get_position="[lon, lat]",
            get_fill_color=[33, 118, 255, 230],
            get_radius=95,
            pickable=True,
        ),
    ]

    if advice.route and advice.route.order:
        by_id = {pick.poi_id: pick for pick in picks}
        path = [[lon, lat]]
        for poi_id in advice.route.order:
            pick = by_id.get(poi_id)
            if pick:
                path.append([pick.lon, pick.lat])
        if advice.route.return_leg:
            path.append([lon, lat])
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": path, "name": "мини-маршрут"}],
                get_path="path",
                get_color=[214, 69, 65, 180],
                get_width=18,
                pickable=False,
            )
        )

    all_points = chosen + me
    center_lat = sum(point["lat"] for point in all_points) / len(all_points)
    center_lon = sum(point["lon"] for point in all_points) / len(all_points)

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=13.2, pitch=0
            ),
            tooltip={"text": "{name}\n{kind}"},
            map_style=None,
        ),
        width="stretch",
    )
    st.caption("🔵 вы · 🔴 рекомендации и маршрут · ⚪ остальные кандидаты, которые прошли фильтры")


def render_cards(advice) -> None:
    picks = advice.answer.picks
    if not picks:
        st.warning(advice.answer.note)
        return

    for number, pick in enumerate(picks, 1):
        with st.container(border=True):
            head, metrics = st.columns([3, 1])
            with head:
                st.markdown(f"**{number}. {pick.name}**  \n`{CATEGORY_RU.get(pick.category, pick.category)}`")
            with metrics:
                st.markdown(f"**{pick.distance_m} м**  \n🚶 {pick.walk_min} мин")

            st.markdown(f"**Почему:** {pick.why}  \n*{SOURCE_LABEL.get(pick.source, pick.source)}*")
            st.markdown(f"**Что делать:** {pick.activity}  \n*идея сгенерирована*")

            left, middle, right = st.columns(3)
            left.markdown(f"⏱ на месте **~{pick.visit_minutes} мин**")
            middle.markdown(HOURS_BADGE.get(pick.hours.get("status"), "⚪ неизвестно"))
            right.markdown(f"уверенность: {CONFIDENCE_BADGE.get(pick.confidence, pick.confidence)}")

            st.caption(f"Часы: {pick.hours.get('detail')}")
            if pick.uncertainties:
                st.caption("⚠️ Не проверено: " + ", ".join(pick.uncertainties))

    if advice.route:
        st.info("🗺 " + describe_route(advice.route, advice.answer.language))

    st.caption(advice.answer.note)


def render_data_panel(advice) -> None:
    used = advice.data_used()
    with st.expander("🔍 Какие данные использованы", expanded=True):
        first, second, third = st.columns(3)
        first.metric("Кандидатов после фильтров", used["candidates_after_filters"])
        second.metric("Ушло в модель", used["candidates_shown_to_model"])
        third.metric("Радиус поиска", f"{used['search_radius_m']} м")

        st.markdown(
            f"""
- **местоположение (округлено до 3 знаков):** `{used['location_rounded'][0]}, {used['location_rounded'][1]}` — {used['location_label']}
- **время:** {used['local_time']} · **бюджет:** {used['time_budget_min']} мин
- **фильтры из запроса:** {', '.join(used['filters'])}
- **датасет:** {used['dataset_size']} POI · {used['dataset_license']}
- **отброшено:** закрыто {used['excluded_closed']} · далеко {used['excluded_too_far']} · не влезает по времени {used['excluded_no_time']}
- **часы неизвестны** у {used['unknown_hours_kept']} кандидатов — помечены, не выдаются за «открыто»
- **погода:** {used['weather']}
- **разбор запроса:** `{used['intent_source']}` · **генерация:** `{used['answer_mode']}` · **модель:** `{used['model']}`
"""
        )
        if used["radius_relaxed"]:
            st.caption("Радиус был расширен автоматически: в исходном слишком мало вариантов.")
        if advice.answer.fallback_reason:
            st.caption(f"Причина шаблонного режима: {advice.answer.fallback_reason}")


def main() -> None:
    st.set_page_config(page_title="Seruen AI — прогулки по Астане", page_icon="🚶", layout="wide")
    init_state()

    st.title("🚶 Seruen AI")
    st.caption(
        "ИИ-помощник по прогулкам в Астане. Поиск по собственному датасету POI (OpenStreetMap, ODbL). "
        "Отвечает 1–3 конкретными местами и показывает, на каких данных построен ответ."
    )

    if not DATASET.exists():
        st.error(f"Нет датасета {DATASET}. Соберите его: `make data`")
        st.stop()

    with st.sidebar:
        st.header("Контекст")
        preset_names = {key: value[0] for key, value in PRESETS.items()}
        preset = st.selectbox(
            "Где вы сейчас",
            options=list(PRESETS),
            format_func=lambda key: preset_names[key],
            index=list(PRESETS).index(st.session_state["preset"]),
        )
        st.session_state["preset"] = preset
        label, lat, lon = PRESETS[preset]

        manual = st.checkbox("Ввести координаты вручную")
        if manual:
            lat = st.number_input("Широта", value=float(lat), format="%.4f")
            lon = st.number_input("Долгота", value=float(lon), format="%.4f")
            label = "указанная точка"

        budget = st.slider("Сколько есть времени, мин", 15, 240, st.session_state["budget"], step=5)
        st.session_state["budget"] = budget

        clock_text = st.text_input(
            "Время (для демо, ЧЧ:ММ)",
            value=st.session_state["clock"] or "",
            placeholder=datetime.now().strftime("%H:%M"),
            help="Пусто — текущее время. Нужно, чтобы показать ночной сценарий днём.",
        )
        st.session_state["clock"] = clock_text.strip() or None

        st.divider()
        st.subheader("Фильтры")
        with_child = st.checkbox("С ребёнком")
        needs_wheelchair = st.checkbox("Без ступеней / коляска")
        group_size = st.number_input("Размер группы", 1, 30, 1)
        indoor_choice = st.radio("Где", ["не важно", "в помещении", "на улице"], horizontal=False)
        language_choice = st.radio("Язык ответа", ["авто", "ru", "kk", "en"], horizontal=True)

        st.divider()
        has_key = bool(os.environ.get("GOOGLE_API_KEY"))
        demo = st.checkbox(
            "Demo-режим (без обращения к модели)",
            value=not has_key,
            help="Ответ собирается только из данных датасета по шаблонам.",
        )
        if not has_key:
            st.caption("⚠️ `GOOGLE_API_KEY` не задан — доступен только demo-режим.")
        use_weather = st.checkbox("Учитывать погоду", value=True)

        st.divider()
        st.caption("Данные: © OpenStreetMap contributors (ODbL), погода — Open-Meteo. "
                   "Координаты не сохраняются, в журнал пишутся округлёнными до 3 знаков.")

    st.subheader("Сценарии из брифа")
    columns = st.columns(len(SCENARIOS))
    for column, scenario in zip(columns, SCENARIOS, strict=True):
        if column.button(scenario["label"], width="stretch"):
            apply_scenario(scenario)
            st.rerun()

    typed = st.chat_input("Напишите, где вы и чего хотите: «я у Байтерека, 40 минут, тихое кафе»")
    query = typed or st.session_state.pop("pending", None)

    for entry in st.session_state["history"][-3:]:
        st.chat_message("user").write(entry)

    if not query:
        st.info(
            "Нажмите один из сценариев выше или напишите запрос. "
            "Система подберёт 1–3 места, построит мини-маршрут и покажет, какие данные использовала."
        )
        return

    st.chat_message("user").write(query)
    st.session_state["history"].append(query)

    overrides = {
        "with_child": with_child,
        "needs_wheelchair": needs_wheelchair,
        "group_size": int(group_size) if group_size > 1 else None,
        "indoor_pref": {"в помещении": "indoor", "на улице": "outdoor"}.get(indoor_choice),
    }

    with st.spinner("Ищу места по датасету…"):
        advice = advise(
            query,
            lat,
            lon,
            budget_min=budget,
            when=resolve_when(st.session_state["clock"]),
            demo=demo,
            use_weather=use_weather,
            origin_label=label,
            dataset_path=DATASET,
            language=None if language_choice == "авто" else language_choice,
            overrides=overrides,
        )

    if advice.weather and advice.weather.get("used"):
        st.caption("🌤 " + weather_mod.describe(advice.weather))

    answer_column, map_column = st.columns([1.15, 1])
    with answer_column:
        render_cards(advice)
    with map_column:
        render_map(advice)

    render_data_panel(advice)


main()
