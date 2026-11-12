"""Command line interface: roaddet run | stats | traffic | cities."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import export, stats
from .cities import CITIES
from .geo import BBox
from .imagery import SOURCES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roaddet", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="detect and measure roads in an area")
    r.add_argument("--city", choices=sorted(CITIES), help="preset area")
    r.add_argument("--bbox", help="west,south,east,north (overrides the preset area)")
    r.add_argument("--source", choices=sorted(SOURCES), help="imagery source")
    r.add_argument("--country", help="ISO country code for default speed limits")
    r.add_argument("--out", type=Path, help="output directory (default data/<city>)")
    r.add_argument("--cache", type=Path, default=Path("cache"))
    r.add_argument("--include", nargs="*", default=[], choices=["service", "pedestrian"])
    r.add_argument("--chunk-km", type=float, default=2.0)
    r.add_argument("--debug", type=int, default=0, help="number of debug strip images to save")
    r.add_argument("--geojson", action="store_true")
    r.add_argument("--reuse-surface", action="store_true",
                   help="reuse the trained model and cached surface rasters of a previous run")
    r.add_argument("--traffic", action="store_true", help="add TomTom traffic (needs TOMTOM_API_KEY)")
    s = sub.add_parser("stats", help="recompute statistics and report from saved results")
    s.add_argument("dir", type=Path)
    t = sub.add_parser("traffic", help="add TomTom traffic to saved results (needs TOMTOM_API_KEY)")
    t.add_argument("dir", type=Path)
    sub.add_parser("cities", help="list preset areas")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if a.cmd == "cities":
        for c in CITIES.values():
            print(f"{c.key:12s} {c.label}  bbox={c.bbox.west},{c.bbox.south},{c.bbox.east},{c.bbox.north}  source={c.source}")
        return
    if a.cmd == "run":
        from . import pipeline
        if not a.city and not a.bbox:
            ap.error("give --city or --bbox")
        c = CITIES.get(a.city)
        bbox = BBox.parse(a.bbox) if a.bbox else c.bbox
        out = a.out or Path("data") / (a.city or "area")
        seg, inter, meta = pipeline.run(
            bbox, out, source=a.source or (c.source if c else "esri"), country=a.country or (c.country if c else "IT"),
            cache=a.cache, include=tuple(a.include), chunk_km=a.chunk_km, debug=a.debug, reuse_surface=a.reuse_surface,
            label=c.label if c and not a.bbox else (a.city or "custom area"))
        if a.traffic:
            from . import traffic
            traffic.enrich(seg)
        seg, inter = export.write(seg, inter, meta, out, geojson=a.geojson)
    else:
        out = a.dir
        seg, inter, meta = export.read(out)
        if a.cmd == "traffic":
            from . import traffic
            seg, inter = export.write(traffic.enrich(seg), inter, meta, out)
    st = stats.write(seg, inter, meta, out)
    w = st["widths_good_fair"]["carriageway_width_m"]
    print(f"{st['network']['segments']} segments, {st['network']['length_km']} km, "
          f"measured {st['network']['measured_share_of_length']:.0%}; carriageway width mean {w.get('mean')} m, "
          f"median {w.get('median')} m. Results in {out}/")


if __name__ == "__main__":
    main()
