"""End-to-end pipeline: OSM network + imagery -> measured segments and intersections."""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import imagery, measure, osm, surface
from .geo import BBox, Projector
from .network import build_network

log = logging.getLogger(__name__)


def _px(geom, grid):
    a = np.array([(g["lon"], g["lat"]) for g in geom])
    c, r = grid.lonlat_to_px(a[:, 0], a[:, 1])
    return np.column_stack([c, r])


def _building_mask(elements, grid, shape):
    polys = []
    for e in elements:
        if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
            polys.append([_px(e["geometry"], grid)])
        elif e["type"] == "relation":
            ways = [m for m in e.get("members", []) if m["type"] == "way" and len(m.get("geometry", [])) >= 3]
            outers = [_px(m["geometry"], grid) for m in ways if m.get("role") != "inner"]
            inners = [_px(m["geometry"], grid) for m in ways if m.get("role") == "inner"]
            polys.extend([o] for o in outers)
            if outers:
                polys[-1] = polys[-1] + inners
    return surface.rasterize_polygons(shape, polys)


class _Chunk:
    """Imagery, building mask and OSM line rasters for one processing tile."""

    def __init__(self, bbox: BBox, src, cache: Path, road_elems, drivable_ids, margin_m=120.0):
        self.bbox = bbox
        big = bbox.expand_m(margin_m)
        self.rgb, self.grid, self.missing = imagery.load_mosaic(src, big, cache)
        self.mpp = self.grid.m_per_px(bbox.center[1])
        h, w = self.rgb.shape[:2]
        self.bld = _building_mask(osm.fetch("buildings", big, cache), self.grid, (h, w))
        self.road_lines, self.all_lines = [], []
        for e in road_elems:
            if e["type"] != "way":
                continue
            b = e.get("bounds")
            if b and (b["maxlon"] < big.west or b["minlon"] > big.east or b["maxlat"] < big.south or b["minlat"] > big.north):
                continue
            px = _px(e["geometry"], self.grid)
            self.all_lines.append(px)
            if e["id"] in drivable_ids and e["tags"].get("tunnel") in (None, "no"):
                self.road_lines.append(px)

    def training_samples(self, n_per_class, block=2048, pad=24):
        """Weakly labelled samples, computed block by block to bound memory use."""
        lab = surface.weak_labels(self.rgb, self.road_lines, self.all_lines, self.bld, self.mpp)
        h, w = lab.shape
        nblk = max(1, (h // block + 1) * (w // block + 1))
        X, y = [], []
        rng = np.random.default_rng(0)
        for r0 in range(0, h, block):
            for c0 in range(0, w, block):
                ra, ca = max(0, r0 - pad), max(0, c0 - pad)
                feat = surface.pixel_features(self.rgb[ra:r0 + block + pad, ca:c0 + block + pad], self.mpp)
                feat = feat[r0 - ra:r0 - ra + block, c0 - ca:c0 - ca + block]
                xs, ys = surface.sample_training(feat, lab[r0:r0 + block, c0:c0 + block],
                                                 max(500, n_per_class // nblk), rng)
                X.append(xs)
                y.append(ys)
        return np.concatenate(X), np.concatenate(y)

    def predict(self, clf, radius_m=32.0, store: Path | None = None):
        """Road probability raster; `store` is a PNG used to save / reuse the result."""
        if store is not None and store.exists():
            prob = cv2.imread(str(store), cv2.IMREAD_GRAYSCALE)
            if prob is not None and prob.shape == self.rgb.shape[:2]:
                self.prob = prob
                return
        mask = surface.rasterize_lines(self.rgb.shape[:2], self.road_lines, int(2 * radius_m / self.mpp))
        mask[self.rgb.sum(axis=2) == 0] = 0
        self.prob = surface.predict(clf, self.rgb, self.mpp, mask)
        if store is not None:
            store.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(store), self.prob)


def _measure_chunk(ch: _Chunk, seg: gpd.GeoDataFrame, idx, proj, params, debug_dir=None, debug_ids=()):
    rows = {}
    for i in idx:
        r = seg.loc[i]
        if r["junction_internal"] or r["tunnel"]:
            rows[i] = {"measured": False, "fail_reason": "junction" if r["junction_internal"] else "tunnel"}
            continue
        mx, my, s, t = measure.strip_maps(r.geometry, proj, ch.grid, params)
        rgb = measure.sample(ch.rgb, mx, my)
        pr = measure.sample(ch.prob, mx, my)
        bl = measure.sample(ch.bld, mx, my, cv2.INTER_NEAREST)
        trim = min(params.end_trim_m, 0.25 * r["length_m"])
        res = measure.measure_strip(rgb, pr, bl, s, t, params,
                                    trim if r["from_degree"] >= 3 else 2.0,
                                    trim if r["to_degree"] >= 3 else 2.0)
        rows[i] = res
        if debug_dir is not None and r["segment_id"] in debug_ids:
            im = measure.debug_strip(rgb, pr, bl, s, t, res, params)
            cv2.imwrite(str(debug_dir / f"{r['segment_id']}.jpg"), im[..., ::-1])
    return rows


_hn = re.compile(r"\d+")


def attach_addresses(seg: gpd.GeoDataFrame, elements, proj, max_dist=80.0) -> None:
    """Civic number range of each segment, from OSM addresses with a matching street name."""
    seg["housenumber_min"] = pd.array([pd.NA] * len(seg), dtype="Int32")
    seg["housenumber_max"] = pd.array([pd.NA] * len(seg), dtype="Int32")
    seg["n_addresses"] = 0
    by_name = defaultdict(list)
    for i, nm in zip(seg.index, seg["name"]):
        if isinstance(nm, str):
            by_name[nm.casefold()].append(i)
    hits = defaultdict(list)
    for e in elements:
        tg = e.get("tags", {})
        m = _hn.search(tg.get("addr:housenumber", ""))
        cand = by_name.get(tg.get("addr:street", "").casefold())
        if not m or not cand:
            continue
        c = e.get("center") or e
        if "lon" not in c:
            continue
        x, y = proj.to_xy(c["lon"], c["lat"])
        d = seg.geometry.loc[cand].distance(gpd.points_from_xy([x], [y])[0])
        if d.min() <= max_dist:
            hits[d.idxmin()].append(int(m.group()))
    for i, nums in hits.items():
        seg.at[i, "housenumber_min"], seg.at[i, "housenumber_max"] = min(nums), max(nums)
        seg.at[i, "n_addresses"] = len(nums)


def _derive(seg: pd.DataFrame) -> None:
    """Derived attributes and quality score from the raw strip measurements."""
    m = seg["measured"].fillna(False).astype(bool)
    seg["measured"] = m
    cw = seg["carriageway_width_m"]
    pk = seg["parking_left_width_m"].fillna(0) + seg["parking_right_width_m"].fillna(0)
    seg["travel_width_m"] = (cw - pk).where(m)
    seg["parking_sides"] = ((seg["parking_left_width_m"] > 0).astype(int)
                            + (seg["parking_right_width_m"] > 0).astype(int)).where(m)
    for side in ("left", "right"):
        w = seg[f"parking_{side}_width_m"]
        seg[f"parking_{side}_type"] = np.select(
            [~m, w == 0, w < 3.2, w < 4.6], [None, "none", "parallel", "diagonal"], "perpendicular")
        seg.loc[~m, f"parking_{side}_type"] = None
    seg["building_to_building_m"] = (seg["facade_left_m"] + seg["facade_right_m"]).where(m)
    sw = seg[["sidewalk_left_width_m", "sidewalk_right_width_m"]]
    seg["sidewalk_sides"] = sw.notna().sum(axis=1).where(m)
    # lanes: OSM when tagged, otherwise from the free travel width (3.5 m per lane, at least one)
    seg["lanes_est"] = np.clip(np.floor(seg["travel_width_m"] / 3.5), 1, 8).where(m).astype("Int16")
    seg["lanes"] = seg["lanes_osm"].astype("Int16").fillna(seg["lanes_est"])
    seg["lanes_source"] = np.where(seg["lanes_osm"].notna(), "osm", np.where(seg["lanes_est"].notna(), "estimated", None))
    canopy = seg[["canopy_edge_left", "canopy_edge_right"]].max(axis=1).fillna(0)
    conf = (seg["block_valid_frac"].fillna(0) * np.exp(-seg["carriageway_width_iqr_m"].fillna(9) / 5.0)
            * (1 - 0.6 * canopy.clip(0, 1)) * (1 - 0.5 * seg["shadow_frac"].fillna(0)))
    seg["confidence"] = conf.where(m, 0.0).round(2)
    seg["quality"] = np.select([~m, seg["confidence"] >= 0.6, seg["confidence"] >= 0.35],
                               ["not_measured", "good", "fair"], "poor")
    seg["road_area_m2"] = (cw * seg["length_m"]).round(0)
    seg["n_width_samples"] = seg["n_width_samples"].astype("Int16")


def _intersection_metrics(inter, seg, ch: _Chunk, proj, done: set) -> None:
    """Size of each intersection in the chunk: mean width of the arms and paved area."""
    lon, lat = inter["lon"].values, inter["lat"].values
    sel = np.flatnonzero(ch.bbox.contains(lon, lat))
    arms = defaultdict(list)
    for a, b, w in zip(seg["from_intersection"], seg["to_intersection"], seg["carriageway_width_m"]):
        if np.isfinite(w):
            for x in (a, b):
                if x:
                    arms[x].append(w)
    for k in sel:
        xid = inter.at[k, "intersection_id"]
        if xid in done:
            continue
        done.add(xid)
        ws = arms.get(xid, [])
        mw = float(np.mean(ws)) if ws else np.nan
        diam = inter.at[k, "node_span_m"] + (mw if ws else 10.0)
        c, r = ch.grid.lonlat_to_px(lon[k], lat[k])
        rad = int(round((diam / 2 + 2.0) / ch.mpp))
        c, r = int(round(c)), int(round(r))
        win = ch.prob[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]
        yy, xx = np.ogrid[:win.shape[0], :win.shape[1]]
        disc = (yy - min(r, rad)) ** 2 + (xx - min(c, rad)) ** 2 <= rad * rad
        inter.at[k, "mean_arm_width_m"] = round(mw, 1) if ws else np.nan
        inter.at[k, "max_arm_width_m"] = round(float(np.max(ws)), 1) if ws else np.nan
        inter.at[k, "approx_diameter_m"] = round(float(diam), 1)
        inter.at[k, "paved_area_m2"] = round(float(((win > 127) & disc).sum()) * ch.mpp ** 2, 0)


def run(bbox: BBox, out_dir: Path, *, source: str = "esri_wayback", country: str = "IT",
        cache: Path = Path("cache"), include: tuple[str, ...] = (), chunk_km: float = 2.0,
        train_chunks: int = 6, debug: int = 0, label: str = "",
        reuse_surface: bool = False):
    """Process an area and return (segments, intersections) GeoDataFrames in EPSG:4326."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = imagery.SOURCES[source]
    proj = Projector(*bbox.center)
    params = measure.StripParams()

    road_elems = []
    seen = set()
    for sub in bbox.split(6.0):  # keep each Overpass request small
        for e in osm.fetch("roads", sub.expand_m(250), cache):
            if (e["type"], e["id"]) not in seen:
                seen.add((e["type"], e["id"]))
                road_elems.append(e)
    seg, inter = build_network(road_elems, proj, bbox, country=country, include=include)
    drivable = set(seg["osm_way_id"])

    chunks = bbox.split(chunk_km)
    which = np.zeros(len(seg), int) - 1
    for k, cb in enumerate(chunks):
        which[np.asarray(cb.contains(seg["mid_lon"].values, seg["mid_lat"].values)) & (which < 0)] = k
    busy = [k for k in range(len(chunks)) if (which == k).any()]

    model_path = out_dir / "surface_model.joblib"
    prob_dir = cache / "prob" / f"{source}_{bbox.key()}"
    if reuse_surface and model_path.exists():
        clf = joblib.load(model_path)
    else:
        reuse_surface = False
        # 1) train the surface classifier on a spread of chunks
        pick = [busy[i] for i in np.unique(np.linspace(0, len(busy) - 1, min(train_chunks, len(busy))).astype(int))]
        X, y = [], []
        for k in pick:
            ch = _Chunk(chunks[k], src, cache, road_elems, drivable)
            if ch.missing > 0.5:
                continue
            xs, ys = ch.training_samples(max(40_000, 240_000 // len(pick)))
            X.append(xs)
            y.append(ys)
        if not X:
            raise RuntimeError("no imagery available for the requested area")
        clf = surface.train(np.concatenate(X), np.concatenate(y))
        clf.weak_label_accuracy_ = float(clf.weak_label_accuracy_)
        joblib.dump(clf, model_path, compress=3)
        del X, y

    # 2) predict + measure chunk by chunk
    debug_dir = None
    debug_ids = set()
    if debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        cand = seg[(seg["length_m"] > 60) & ~seg["junction_internal"]]
        debug_ids = set(cand.sample(min(debug, len(cand)), random_state=0)["segment_id"])
    results, done_x, missing = {}, set(), []
    for col in ("mean_arm_width_m", "max_arm_width_m", "approx_diameter_m", "paved_area_m2"):
        inter[col] = np.nan
    seg["carriageway_width_m"] = np.nan
    for n, k in enumerate(busy):
        tc = time.time()
        ch = _Chunk(chunks[k], src, cache, road_elems, drivable)
        missing.append(ch.missing)
        store = prob_dir / f"{chunks[k].key()}.png"
        if not reuse_surface:
            store.unlink(missing_ok=True)
        ch.predict(clf, store=store)
        idx = seg.index[which == k]
        rows = _measure_chunk(ch, seg, idx, proj, params, debug_dir, debug_ids)
        results.update(rows)
        for i, r in rows.items():
            if r.get("measured"):
                seg.at[i, "carriageway_width_m"] = r["carriageway_width_m"]
        _intersection_metrics(inter, seg, ch, proj, done_x)
        log.info("chunk %d/%d: %d segments, %.0fs", n + 1, len(busy), len(idx), time.time() - tc)
        del ch
    res = pd.DataFrame.from_dict(results, orient="index").reindex(seg.index)
    cols = ["measured", "fail_reason", "carriageway_width_m", "carriageway_width_iqr_m",
            "carriageway_width_min_m", "carriageway_width_mean_m", "carriageway_width_median_m",
            "carriageway_width_max_m", "n_width_samples", "axis_offset_m",
            "parking_left_width_m", "parking_right_width_m", "parking_left_share", "parking_right_share",
            "parking_left_occupancy", "parking_right_occupancy", "sidewalk_left_width_m",
            "sidewalk_right_width_m", "facade_left_m", "facade_right_m", "canopy_edge_left",
            "canopy_edge_right", "canopy_frac", "shadow_frac", "road_prob_axis", "block_valid_frac"]
    for c in cols:
        seg[c] = res[c] if c in res else np.nan
    _derive(seg)

    try:
        addr = []
        for sub in bbox.split(6.0):
            addr += osm.fetch("addresses", sub.expand_m(60), cache)
        attach_addresses(seg, addr, proj)
    except RuntimeError as exc:  # addresses are optional
        log.warning("addresses not available: %s", exc)

    meta = {
        "label": label, "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
        "area_km2": round(bbox.area_km2(), 2), "imagery": src.attribution, "imagery_source": source,
        "ground_resolution_m": round(imagery.ground_resolution(src, bbox.center[1]), 3),
        "missing_imagery_frac": round(float(np.mean(missing)), 3) if missing else None,
        "osm": "OpenStreetMap contributors (ODbL)",
        "classifier_weak_label_accuracy": round(clf.weak_label_accuracy_, 3),
        "crs_metric": proj.crs.to_string(), "runtime_s": round(time.time() - t0),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return seg.to_crs(4326), inter.to_crs(4326), meta
