"""Normalisation of OpenStreetMap road tags into analysis-friendly attributes."""
from __future__ import annotations

import re

DRIVABLE = [
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]
OPTIONAL = ["service", "pedestrian"]

# Urban default speed limit by country code (km/h) used when OSM has no maxspeed.
URBAN_DEFAULT_KMH = {"IT": 50, "NL": 50, "FR": 50, "DE": 50, "ES": 50, "CH": 50, "AT": 50}

_num = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _float(v):
    if v is None:
        return None
    m = _num.search(str(v))
    if not m:
        return None
    x = float(m.group().replace(",", "."))
    s = str(v).lower()
    if "ft" in s or "'" in s:
        x *= 0.3048
    return x


def _int(v):
    x = _float(v)
    return int(x) if x is not None and 0 < x < 20 else None


def parse_maxspeed(tags: dict, country: str = "IT"):
    """Return (km/h or None, source)."""
    v = tags.get("maxspeed")
    if v:
        s = str(v).strip().lower()
        if s in ("walk", "walking"):
            return 5, "osm"
        if s.endswith(":urban") or s == "urban":
            return URBAN_DEFAULT_KMH.get(country, 50), "osm_implicit"
        if s.endswith(":living_street"):
            return 20, "osm_implicit"
        x = _float(s)
        if x is not None and s not in ("none", "signals", "variable"):
            return int(round(x * 1.609344 if "mph" in s else x)), "osm"
    hw = tags.get("highway", "")
    if hw in ("motorway", "trunk", "motorway_link", "trunk_link"):
        return None, "unknown"
    if hw == "living_street":
        return 20, "default"
    if hw == "pedestrian":
        return None, "unknown"
    return URBAN_DEFAULT_KMH.get(country, 50), "default"


def parse_oneway(tags: dict) -> str:
    """Return 'forward', 'backward' (relative to the digitised geometry) or 'both'."""
    v = str(tags.get("oneway", "")).lower()
    if v in ("yes", "true", "1"):
        return "forward"
    if v in ("-1", "reverse"):
        return "backward"
    if v in ("no", "false", "0", "reversible", "alternating"):
        return "both"
    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") in ("motorway", "motorway_link"):
        return "forward"
    return "both"


def parse_sidewalk(tags: dict):
    """Return one of both / left / right / no / separate or None when untagged."""
    left = right = None

    def setv(side, val):
        nonlocal left, right
        val = str(val).lower()
        val = {"none": "no", "yes": "yes", "no": "no", "separate": "separate"}.get(val)
        if val is None:
            return
        if side in ("left", "both"):
            left = val
        if side in ("right", "both"):
            right = val

    v = str(tags.get("sidewalk", "")).lower()
    if v in ("both", "left", "right"):
        setv("both", "no")
        setv(v, "yes")
    elif v in ("no", "none", "separate"):
        setv("both", v)
    elif v == "yes":
        setv("both", "yes")
    for side in ("both", "left", "right"):
        if f"sidewalk:{side}" in tags:
            setv(side, tags[f"sidewalk:{side}"])
    if left is None and right is None:
        return None
    if left == "separate" or right == "separate":
        return "separate"
    l, r = left == "yes", right == "yes"
    return "both" if l and r else "left" if l else "right" if r else "no"


_PARK_NO = {"no", "no_parking", "no_stopping", "no_standing", "fire_lane", "none"}
_PARK_ORIENT = {"parallel", "diagonal", "perpendicular"}


def parse_parking(tags: dict):
    """Return (left, right), each parallel / diagonal / perpendicular / yes / no / None."""
    out = {"left": None, "right": None}

    def setv(side, val):
        for s in (("left", "right") if side == "both" else (side,)):
            out[s] = val

    for side in ("both", "left", "right"):
        # legacy scheme: parking:lane:<side>=parallel|no_parking|...
        v = tags.get(f"parking:lane:{side}")
        if v:
            v = str(v).lower()
            setv(side, v if v in _PARK_ORIENT else "no" if v in _PARK_NO else "yes" if v != "separate" else None)
        # current scheme: parking:<side>=lane|street_side|on_kerb|no + parking:<side>:orientation
        v = tags.get(f"parking:{side}")
        if v:
            v = str(v).lower()
            if v in _PARK_NO:
                setv(side, "no")
            elif v in ("lane", "street_side", "on_kerb", "half_on_kerb", "shoulder", "yes"):
                o = str(tags.get(f"parking:{side}:orientation", "")).lower()
                setv(side, o if o in _PARK_ORIENT else "yes")
    return out["left"], out["right"]


def road_attributes(tags: dict, country: str = "IT") -> dict:
    ms, ms_src = parse_maxspeed(tags, country)
    pl, pr = parse_parking(tags)
    cyc = any(k.startswith("cycleway") and str(v).lower() not in ("no", "none", "separate")
              for k, v in tags.items())
    return {
        "name": tags.get("name") or tags.get("ref") or None,
        "ref": tags.get("ref"),
        "highway": tags.get("highway"),
        "oneway": parse_oneway(tags),
        "lanes_osm": _int(tags.get("lanes")),
        "maxspeed_kmh": ms,
        "maxspeed_source": ms_src,
        "width_osm_m": _float(tags.get("width:carriageway") or tags.get("width")),
        "sidewalk_osm": parse_sidewalk(tags),
        "parking_osm_left": pl,
        "parking_osm_right": pr,
        "surface_osm": tags.get("surface"),
        "lit_osm": tags.get("lit"),
        "cycleway_osm": cyc,
        "bus_lane_osm": any(k.startswith(("busway", "lanes:bus", "lanes:psv")) for k in tags),
        "bridge": tags.get("bridge") not in (None, "no"),
        "tunnel": tags.get("tunnel") not in (None, "no") or tags.get("covered") == "yes",
        "roundabout": tags.get("junction") in ("roundabout", "circular"),
    }
