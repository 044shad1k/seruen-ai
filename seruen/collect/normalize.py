"""raw Overpass dump -> data/pois.json (the dataset the RAG retrieves over).

Stdlib only, same reason as overpass.py.

Usage:
    python -m seruen.collect.normalize [--raw data/raw] [--out data/pois.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEDUP_RADIUS_M = 50

# (osm key, osm value) -> our category
CATEGORY_MAP = {
    ("leisure", "park"): "park",
    ("leisure", "garden"): "park",
    ("leisure", "playground"): "playground",
    ("leisure", "sports_centre"): "sport",
    ("leisure", "ice_rink"): "sport",
    ("leisure", "water_park"): "sport",
    ("amenity", "cafe"): "cafe",
    ("amenity", "ice_cream"): "cafe",
    ("amenity", "restaurant"): "restaurant",
    ("amenity", "fast_food"): "restaurant",
    ("amenity", "bar"): "restaurant",
    ("amenity", "pub"): "restaurant",
    ("amenity", "library"): "library",
    ("amenity", "theatre"): "theatre",
    ("amenity", "cinema"): "theatre",
    ("amenity", "arts_centre"): "theatre",
    ("tourism", "museum"): "museum",
    ("tourism", "gallery"): "museum",
    ("tourism", "attraction"): "landmark",
    ("tourism", "artwork"): "landmark",
    ("tourism", "zoo"): "landmark",
    ("tourism", "theme_park"): "landmark",
    ("tourism", "viewpoint"): "viewpoint",
    ("historic", "monument"): "landmark",
    ("historic", "memorial"): "landmark",
    ("shop", "mall"): "mall",
}

CATEGORY_LABEL_RU = {
    "park": "Парк",
    "cafe": "Кафе",
    "restaurant": "Ресторан",
    "museum": "Музей",
    "landmark": "Достопримечательность",
    "viewpoint": "Смотровая площадка",
    "playground": "Детская площадка",
    "mall": "Торговый центр",
    "theatre": "Театр или кинотеатр",
    "library": "Библиотека",
    "sport": "Спортивный объект",
    "other": "Место",
}

# osm value -> russian noun, used to build a readable description
SUBTYPE_RU = {
    "cafe": "кафе",
    "ice_cream": "кафе-мороженое",
    "restaurant": "ресторан",
    "fast_food": "фастфуд",
    "bar": "бар",
    "pub": "паб",
    "library": "библиотека",
    "theatre": "театр",
    "cinema": "кинотеатр",
    "arts_centre": "центр искусств",
    "museum": "музей",
    "gallery": "галерея",
    "attraction": "достопримечательность",
    "artwork": "арт-объект",
    "zoo": "зоопарк",
    "theme_park": "парк развлечений",
    "viewpoint": "смотровая площадка",
    "monument": "памятник",
    "memorial": "мемориал",
    "mall": "торговый центр",
    "park": "парк",
    "garden": "сад",
    "playground": "детская площадка",
    "sports_centre": "спортивный центр",
    "ice_rink": "ледовый каток",
    "water_park": "аквапарк",
}

CUISINE_RU = {
    "regional": "местная",
    "kazakh": "казахская",
    "asian": "азиатская",
    "chinese": "китайская",
    "japanese": "японская",
    "sushi": "суши",
    "korean": "корейская",
    "italian": "итальянская",
    "pizza": "пицца",
    "pasta": "паста",
    "georgian": "грузинская",
    "turkish": "турецкая",
    "uzbek": "узбекская",
    "indian": "индийская",
    "american": "американская",
    "burger": "бургеры",
    "steak_house": "стейки",
    "barbecue": "гриль",
    "chicken": "курица",
    "kebab": "шашлык и кебаб",
    "mexican": "мексиканская",
    "french": "французская",
    "european": "европейская",
    "international": "интернациональная",
    "russian": "русская",
    "seafood": "морепродукты",
    "fish": "рыба",
    "bakery": "выпечка",
    "dessert": "десерты",
    "ice_cream": "мороженое",
    "coffee_shop": "кофейня",
    "coffee": "кофе",
    "tea": "чай",
    "breakfast": "завтраки",
    "sandwich": "сэндвичи",
    "vegetarian": "вегетарианская",
    "vegan": "веганская",
    "noodle": "лапша",
    "soup": "супы",
    "local": "местная",
}

SPORT_RU = {
    "ice_skating": "катание на коньках",
    "ice_hockey": "хоккей",
    "swimming": "бассейн",
    "fitness": "фитнес",
    "billiards": "бильярд",
    "bowling": "боулинг",
    "climbing": "скалолазание",
    "tennis": "теннис",
    "basketball": "баскетбол",
    "soccer": "футбол",
    "football": "футбол",
    "skateboard": "скейтборд",
    "yoga": "йога",
    "multi": "несколько видов спорта",
    "gymnastics": "гимнастика",
    "martial_arts": "единоборства",
    "boxing": "бокс",
    "dance": "танцы",
}

# categories that are reliably indoor / outdoor; None = we do not know
INDOOR_BY_SUBTYPE = {
    "cafe": True,
    "ice_cream": True,
    "restaurant": True,
    "fast_food": True,
    "bar": True,
    "pub": True,
    "library": True,
    "theatre": True,
    "cinema": True,
    "arts_centre": True,
    "museum": True,
    "gallery": True,
    "mall": True,
    "sports_centre": True,
    "park": False,
    "garden": False,
    "playground": False,
    "viewpoint": False,
    "monument": False,
    "memorial": False,
    "artwork": False,
}


def _norm_name(name: str) -> str:
    """Casefold + strip punctuation, for dedup comparisons only."""
    text = unicodedata.normalize("NFKD", name).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(math.sqrt(a))


def _pick_category(tags: dict) -> tuple[str, str | None]:
    """Our category plus the raw OSM subtype it came from."""
    for key in ("tourism", "historic", "leisure", "amenity", "shop"):
        value = tags.get(key)
        if value and (key, value) in CATEGORY_MAP:
            return CATEGORY_MAP[(key, value)], value
    return "other", None


def _coords(element: dict) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def _build_tags(tags: dict, subtype: str | None) -> list[str]:
    """Searchable keywords (russian where possible) for BM25."""
    out: list[str] = []
    if subtype:
        out.append(SUBTYPE_RU.get(subtype, subtype))
    for raw_cuisine in str(tags.get("cuisine", "")).split(";"):
        raw_cuisine = raw_cuisine.strip()
        if raw_cuisine:
            out.append(CUISINE_RU.get(raw_cuisine, raw_cuisine.replace("_", " ")))
    for raw_sport in str(tags.get("sport", "")).split(";"):
        raw_sport = raw_sport.strip()
        if raw_sport:
            out.append(SPORT_RU.get(raw_sport, raw_sport.replace("_", " ")))
    if tags.get("outdoor_seating") == "yes":
        out.append("летняя терраса")
    if tags.get("indoor_seating") == "yes":
        out.append("места внутри")
    if tags.get("internet_access") in {"wlan", "yes", "wifi"}:
        out.append("wi-fi")
    if tags.get("takeaway") in {"yes", "only"}:
        out.append("с собой")
    if tags.get("air_conditioning") == "yes":
        out.append("кондиционер")
    if tags.get("smoking") in {"no", "outside"}:
        out.append("без курения")
    if tags.get("opening_hours") in {"24/7", "24x7"}:
        out.append("круглосуточно")
    if tags.get("playground") or tags.get("leisure") == "playground":
        out.append("для детей")
    if tags.get("brand"):
        out.append(str(tags["brand"]))
    if tags.get("operator"):
        out.append(str(tags["operator"]))
    if tags.get("wheelchair") == "yes":
        out.append("доступно на коляске")
    if tags.get("fee") == "no":
        out.append("бесплатно")
    seen, unique = set(), []
    for item in out:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _build_description(tags: dict, category: str, subtype: str | None, keywords: list[str]) -> str:
    """One or two readable russian sentences derived purely from OSM tags."""
    noun = SUBTYPE_RU.get(subtype or "", CATEGORY_LABEL_RU.get(category, "место").lower())
    sentences = [noun.capitalize() + " в Астане."]

    details: list[str] = []
    cuisines = [
        CUISINE_RU.get(c.strip(), c.strip().replace("_", " "))
        for c in str(tags.get("cuisine", "")).split(";")
        if c.strip()
    ]
    if cuisines:
        details.append("кухня — " + ", ".join(cuisines))
    sports = [
        SPORT_RU.get(s.strip(), s.strip().replace("_", " "))
        for s in str(tags.get("sport", "")).split(";")
        if s.strip()
    ]
    if sports:
        details.append("занятия — " + ", ".join(sports))
    if tags.get("outdoor_seating") == "yes":
        details.append("есть летняя терраса")
    if tags.get("internet_access") in {"wlan", "yes", "wifi"}:
        details.append("отмечен wi-fi")
    if tags.get("takeaway") in {"yes", "only"}:
        details.append("можно взять с собой")
    if tags.get("fee") == "no":
        details.append("вход бесплатный")
    if tags.get("wheelchair") == "yes":
        details.append("доступно на коляске")
    elif tags.get("wheelchair") == "limited":
        details.append("доступность на коляске ограничена")
    if details:
        sentences.append(("; ".join(details)).capitalize() + ".")

    street = tags.get("addr:street")
    house = tags.get("addr:housenumber")
    if street:
        sentences.append(f"Адрес: {street}{' ' + house if house else ''}.")
    elif keywords and len(sentences) == 1:
        sentences.append("Особенности по данным OSM: " + ", ".join(keywords[:4]) + ".")
    return " ".join(sentences)


def _indoor(tags: dict, subtype: str | None) -> bool | None:
    if tags.get("indoor") in {"yes", "room"}:
        return True
    if tags.get("indoor") == "no":
        return False
    return INDOOR_BY_SUBTYPE.get(subtype or "")


def _website(tags: dict) -> str | None:
    for key in ("website", "contact:website", "url", "contact:instagram"):
        if tags.get(key):
            return str(tags[key])
    return None


def _phone(tags: dict) -> str | None:
    for key in ("phone", "contact:phone", "contact:mobile"):
        if tags.get(key):
            return str(tags[key])
    return None


def normalize_element(element: dict) -> dict | None:
    """One Overpass element -> one POI record, or None if unusable."""
    tags = element.get("tags") or {}
    name = (tags.get("name") or "").strip()
    if not name:
        return None
    position = _coords(element)
    if position is None:
        return None
    lat, lon = position

    category, subtype = _pick_category(tags)
    keywords = _build_tags(tags, subtype)
    osm_id = f"{element['type']}/{element['id']}"
    osm_url = f"https://www.openstreetmap.org/{element['type']}/{element['id']}"

    evidence = [{"field": "name", "source_url": osm_url}, {"field": "category", "source_url": osm_url}]
    if tags.get("opening_hours"):
        evidence.append({"field": "hours_raw", "source_url": osm_url})
    if tags.get("wheelchair"):
        evidence.append({"field": "wheelchair", "source_url": osm_url})

    return {
        "id": osm_id,
        "name": {
            "ru": tags.get("name:ru") or name,
            "kk": tags.get("name:kk") or name,
            "en": tags.get("name:en") or name,
        },
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "category": category,
        "osm_subtype": subtype,
        "tags": keywords,
        "description": _build_description(tags, category, subtype, keywords),
        "description_source": "osm_tags",
        "hours_raw": tags.get("opening_hours"),
        "wheelchair": tags.get("wheelchair") if tags.get("wheelchair") in {"yes", "no", "limited"} else "unknown",
        "indoor": _indoor(tags, subtype),
        "website": _website(tags),
        "phone": _phone(tags),
        "wikidata": tags.get("wikidata"),
        "source": {"url": osm_url, "license": "ODbL", "fetched_at": None},
        "evidence": evidence,
    }


def dedup(pois: list[dict], radius_m: float = DEDUP_RADIUS_M) -> tuple[list[dict], int]:
    """Drop same-name POIs sitting within `radius_m` of each other.

    Keeps the record with the most filled-in fields (richer OSM tagging).
    """
    def richness(poi: dict) -> int:
        score = len(poi["tags"])
        for field in ("hours_raw", "website", "phone", "wikidata"):
            if poi.get(field):
                score += 2
        if poi.get("wheelchair") != "unknown":
            score += 1
        return score

    ordered = sorted(pois, key=richness, reverse=True)
    buckets: dict[str, list[dict]] = {}
    kept: list[dict] = []
    removed = 0
    for poi in ordered:
        key = _norm_name(poi["name"]["ru"])
        neighbours = buckets.setdefault(key, [])
        if any(_haversine_m(poi["lat"], poi["lon"], o["lat"], o["lon"]) <= radius_m for o in neighbours):
            removed += 1
            continue
        neighbours.append(poi)
        kept.append(poi)
    kept.sort(key=lambda p: p["name"]["ru"].casefold())
    return kept, removed


def latest_raw(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("overpass_*.json"))
    if not files:
        raise FileNotFoundError(f"no overpass dumps in {raw_dir} — run seruen.collect.overpass first")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize raw Overpass data into pois.json")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/pois.json")
    parser.add_argument("--report", default="data/quality_report.json")
    parser.add_argument("--meta", default="data/metadata.json")
    args = parser.parse_args(argv)

    raw_path = latest_raw(Path(args.raw))
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])

    fetched_at = None
    meta_path = Path(args.meta)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = (meta.get("latest") or {}).get("fetched_at")
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    pois: list[dict] = []
    dropped_unnamed = 0
    dropped_no_coords = 0
    for element in elements:
        tags = element.get("tags") or {}
        if not (tags.get("name") or "").strip():
            dropped_unnamed += 1
            continue
        if _coords(element) is None:
            dropped_no_coords += 1
            continue
        record = normalize_element(element)
        if record is None:
            continue
        record["source"]["fetched_at"] = fetched_at
        pois.append(record)

    pois, dropped_duplicates = dedup(pois)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pois, ensure_ascii=False, indent=1), encoding="utf-8")

    by_category = Counter(poi["category"] for poi in pois)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": str(raw_path),
        "raw_elements": len(elements),
        "pois_total": len(pois),
        "dropped_unnamed": dropped_unnamed,
        "dropped_no_coords": dropped_no_coords,
        "dropped_duplicates": dropped_duplicates,
        "with_hours": sum(1 for p in pois if p["hours_raw"]),
        "with_hours_pct": round(100 * sum(1 for p in pois if p["hours_raw"]) / max(1, len(pois)), 1),
        "with_website": sum(1 for p in pois if p["website"]),
        "with_phone": sum(1 for p in pois if p["phone"]),
        "with_wikidata": sum(1 for p in pois if p["wikidata"]),
        "wheelchair_known": sum(1 for p in pois if p["wheelchair"] != "unknown"),
        "wheelchair_yes": sum(1 for p in pois if p["wheelchair"] == "yes"),
        "name_ru_localized": sum(1 for p in pois if p["name"]["ru"] != p["name"]["en"]),
        "indoor_known": sum(1 for p in pois if p["indoor"] is not None),
        "by_category": dict(by_category.most_common()),
        "license": "ODbL 1.0 — © OpenStreetMap contributors",
        "known_limitations": [
            "часы работы есть не у всех объектов — такие помечаются как unknown, а не «открыто»",
            "розетки, уровень шума и наличие пеленальной комнаты в OSM почти не размечены",
            "координаты way/relation приведены к одной центральной точке (out center)",
        ],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[normalize] {len(pois)} POIs -> {args.out}")
    print(f"[normalize] hours: {report['with_hours']} ({report['with_hours_pct']}%), duplicates removed: {dropped_duplicates}")
    print(f"[normalize] categories: {dict(by_category.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
