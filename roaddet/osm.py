"""OpenStreetMap download through the Overpass API (with mirrors and a disk cache)."""
from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path

import requests

from .geo import BBox

log = logging.getLogger(__name__)

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

QUERIES = {
    "roads": """
        (way["highway"]({b});
         node["highway"~"^(traffic_signals|crossing|mini_roundabout|stop|give_way)$"]({b});
         node["crossing"]({b}););
        out body geom;""",
    "buildings": """
        (way["building"]({b});
         relation["building"]["type"="multipolygon"]({b}););
        out body geom;""",
    "addresses": """
        (node["addr:housenumber"]["addr:street"]({b});
         way["addr:housenumber"]["addr:street"]({b}););
        out tags center;""",
}

_good_mirror: list[str] = []


def _overpass(query: str) -> dict:
    last = None
    mirrors = _good_mirror + [m for m in OVERPASS_MIRRORS if m not in _good_mirror]
    for rnd in range(3):
        for url in mirrors:
            try:
                r = requests.post(url, data={"data": query}, timeout=(8, 300),
                                  headers={"User-Agent": "roaddet/0.1"})
                if r.status_code == 200:
                    data = r.json()
                    if "elements" in data:
                        if url not in _good_mirror:
                            _good_mirror.insert(0, url)
                        return data
                last = f"{url}: HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last = f"{url}: {type(exc).__name__}"
            log.debug("overpass failure %s", last)
        time.sleep(5 * (rnd + 1))
    raise RuntimeError(f"all Overpass mirrors failed ({last})")


def fetch(kind: str, bbox: BBox, cache: Path, refresh: bool = False) -> list[dict]:
    """Return OSM elements of the given kind (roads | buildings | addresses) in bbox."""
    path = cache / "osm" / f"{kind}_{bbox.key()}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)["elements"]
    b = f"{bbox.south:.6f},{bbox.west:.6f},{bbox.north:.6f},{bbox.east:.6f}"
    query = "[out:json][timeout:240][maxsize:1073741824];" + QUERIES[kind].format(b=b)
    t = time.time()
    data = _overpass(query)
    log.info("overpass %s: %d elements in %.1fs", kind, len(data["elements"]), time.time() - t)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return data["elements"]
