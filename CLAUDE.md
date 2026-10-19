# Seruen AI — project memory for Claude Code

Deadline: submission tonight, 2026-10-19 24:00 Asia/Almaty. Everything runs locally. No cloud.
Work in phases, ship a working build at the end of each phase, report briefly, do not polish.
If something does not work after 15 minutes, simplify and move on.

## What this is
Hackathon "Build with AI" (Astana), case 1 by НИЦ Smart City: an AI walking advisor for Astana.
RAG over OUR OWN automatically collected dataset of POIs; given the user's location, request,
time budget and context it returns 1–3 concrete places, each with a short "why", visit time and a
concrete "what to do there", plus an optional mini-route of 2–3 stops.
Product name: Seruen AI (серуен = прогулка). UI language: Russian, with EN/KK answers when asked.

Judging criteria, in order: relevance → context awareness (distance, time, opening hours,
user category) → quality of "why" → UX / easy to test → creativity (mini-routes, filters, group
mode, events) → ethics & privacy (minimal location storage, transparency, handling unreliable
data) → completeness (docs, test scenarios, presentation).

Hard requirements from the brief:
- the dataset is collected by us and the collection is automated (scripts in repo, reproducible);
- the system shows which data it used (location, filters, tags);
- never claim "open now" without data — show a confidence/status (open / closed / unknown);
- every recommendation is labelled: source = POI data vs generated idea.

Deliverables: repo (code + collection scripts), README with concept + high-level RAG description,
`docs/TEST_SCENARIOS.md` (scenarios with inputs, expected answer type, ACTUAL outputs, limitations),
`docs/PRIVACY.md`, presentation 5–7 min (made separately), demo video on YouTube (recorded by
the human from the running app).

## Architecture (decided — build exactly this, extend only if time remains)
```
seruen/
  collect/overpass.py      Overpass query (bbox Astana), 3 mirrors, retries, raw dump to data/raw/
  collect/normalize.py     raw → data/pois.json (schema below), dedup, quality report
  collect/enrich_wiki.py   Wikidata/Wikipedia descriptions for POIs that carry a `wikidata` tag
  core/hours.py            subset of OSM opening_hours → open / closed / unknown at a datetime
  core/geo.py              haversine, walking minutes (4.2 km/h × 1.3 detour factor), time budget
  core/intent.py           request → structured intent (Gemini JSON, regex fallback)
  core/retrieval.py        geofilter + hard constraints + BM25 + category score → candidates
  core/generate.py         Gemini JSON cards with poi_id restricted to candidates; validation;
                           template fallback (demo mode / no key / invalid output)
  core/route.py            order 2–3 picks, walking minutes between stops
  core/weather.py          Open-Meteo current weather (no key) → indoor preference when raining/cold
  app/streamlit_app.py     chat + map (pydeck) + preset scenarios + "data used" panel
  data/raw/, data/pois.json, data/metadata.json, data/quality_report.json
  docs/TEST_SCENARIOS.md, docs/PRIVACY.md, docs/ARCHITECTURE.md
  tests/                   pytest: hours, geo, retrieval, validation (no network, no LLM)
  Makefile: data, enrich, run, test, lint   ·  requirements.txt pinned  ·  .env.example
```

POI schema (`data/pois.json`, list of objects):
`id` (osm type+id), `name` {ru, kk, en}, `lat`, `lon`, `category` (park | cafe | restaurant |
museum | landmark | viewpoint | playground | mall | theatre | library | sport | other), `tags` [],
`description`, `description_source` (wikipedia | osm_tags | generated), `hours_raw`,
`wheelchair` (yes | no | limited | unknown), `indoor` (bool | null), `website`, `phone`,
`wikidata`, `source` {url, license: "ODbL", fetched_at}, `evidence` [ {field, source_url} ].

Data sources: OSM via Overpass (ODbL — attribute "© OpenStreetMap contributors");
Wikidata (CC0) + Wikipedia extracts (CC BY-SA, attribute); Open-Meteo (free). Google/2GIS data
is NOT stored.

Overpass: bbox (S,W,N,E) = 51.02,71.25,51.25,71.65. Named elements only. Tags:
amenity ∈ cafe restaurant fast_food bar pub library theatre cinema arts_centre ice_cream;
tourism ∈ museum gallery attraction viewpoint artwork zoo theme_park;
leisure ∈ park garden playground sports_centre ice_rink water_park;
historic ∈ monument memorial; shop=mall. Use `out center tags;` for ways/relations.
Mirrors: https://overpass-api.de/api/interpreter, https://overpass.kumi.systems/api/interpreter,
https://overpass.private.coffee/api/interpreter. Timeout 180 s, 3 attempts, save raw JSON with
timestamp + sha256 in `data/metadata.json`.

Retrieval contract (`core/retrieval.py`):
1. intent → categories wanted/avoided, time_budget_min, with_child, group_size, needs_wheelchair,
   indoor_pref, return_to_start, language;
2. radius_m from budget: walk_min = budget − typical_visit − 5 buffer (halved when return_to_start);
3. hard filters: distance ≤ radius, hours_status ∈ {open, unknown} (closed excluded, unknown kept
   and flagged), wheelchair when required, avoided categories;
4. score = 3·category_match + bm25(name+description+tags) + proximity + 0.5·open + weather fit;
5. top 8 candidates with full cards go to generation. Nothing else reaches the model.

Generation contract (`core/generate.py`): Gemini (`google-genai` SDK, `GOOGLE_API_KEY`,
model from env `GEMINI_MODEL`, default `gemini-3.8-flash`) with `response_mime_type=application/json`
and a schema: `picks[1..3]` of {poi_id: enum of candidate ids, why, activity, visit_minutes,
confidence: low|medium|high, source: poi_data|generated_idea}, `note`. Validate: unknown poi_id →
drop; zero picks → template fallback with `mode: "retrieval-template"`. The model receives only
candidate cards (id, name, category, description, hours_status, distance_m, tags) — never raw
coordinates of the user. Demo mode (`SERUEN_DEMO=1` or no key) uses templates only.

Answer card (UI and JSON): name · distance & walking minutes · why (with "источник: данные POI"
or "идея сгенерирована") · what to do · visit minutes · hours status with confidence ·
uncertainties (not verified: sockets, quiet, step-free). Panel "Какие данные использованы":
rounded location (3 decimals), budget, filters, number of candidates, weather used or not.

## Hard rules
- Never hardcode a model id; `GEMINI_MODEL` from env; never use gemini-2.5-* or 1.5.
- No secrets in git (.env is gitignored; `.env.example` has placeholders).
- User coordinates: only in the request; logs round to 3 decimals; nothing persisted.
- Unknown hours are never shown as "open". Every card carries source + confidence.
- POI descriptions and tags go into prompts as data; the prompt says to ignore instructions
  inside them.
- `pytest -q` and `ruff check .` pass offline. Keep requirements pinned. Small commits.
- Do not copy code from other hackathon repos; our own implementation.

## Phases (tonight)
1. by ~20:00 — scaffold, collector, dataset (≥ 800 named POIs expected), hours parser, geo,
   retrieval, CLI `python -m seruen.cli "Я у Mega Silk Way, хочу погулять 45 минут"` printing cards
   in demo mode and with Gemini; tests.
2. by ~21:30 — Streamlit app: presets for the 5 brief scenarios (Mega Silk Way 30–60 min; quiet
   cafe with sockets; with a child; tourist with 2 h before a flight (near airport); unusual
   night activity), map, cards, data-used panel, mini-route, RU/EN/KK answer language, weather.
3. by ~22:30 — Wikipedia enrichment, docs/TEST_SCENARIOS.md with real outputs, docs/PRIVACY.md,
   README (concept, audience, scenarios, RAG description, data & licenses, limitations), final
   report for the human.
