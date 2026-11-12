"""Build the road graph: street segments between junction nodes, and intersection objects."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from .geo import BBox, Projector
from .tags import DRIVABLE, OPTIONAL, road_attributes

log = logging.getLogger(__name__)


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_network(elements: list[dict], proj: Projector, bbox: BBox, *, country: str = "IT",
                  include: tuple[str, ...] = (), cluster_m: float = 15.0):
    """Return (segments GeoDataFrame, intersections GeoDataFrame), both in the metric CRS.

    A segment is the part of an OSM way between two consecutive break nodes (nodes shared
    with another drivable way, or way ends). Intersections are clusters of break nodes with
    three or more arms; nodes of one roundabout, or joined by very short links (as in dual
    carriageway crossings), are merged into a single intersection object.
    """
    classes = set(DRIVABLE) | (set(include) & set(OPTIONAL))
    ways = [e for e in elements if e["type"] == "way" and e.get("tags", {}).get("highway") in classes
            and e["tags"].get("area") != "yes" and len(e.get("nodes", [])) >= 2]
    use = Counter()
    for w in ways:
        use.update(w["nodes"])
        use[w["nodes"][0]] += 1  # way ends always break
        use[w["nodes"][-1]] += 1
    rows, node_ll = [], {}
    for w in ways:
        nodes, geom = w["nodes"], w["geometry"]
        attrs = road_attributes(w["tags"], country)
        start, idx = 0, 0
        for i in range(1, len(nodes)):
            if use[nodes[i]] >= 2 or i == len(nodes) - 1:
                pts = [(g["lon"], g["lat"]) for g in geom[start:i + 1]]
                node_ll[nodes[start]], node_ll[nodes[i]] = pts[0], pts[-1]
                rows.append({"segment_id": f"{w['id']}_{idx}", "osm_way_id": w["id"],
                             "from_node": nodes[start], "to_node": nodes[i], "_ll": pts, **attrs})
                start, idx = i, idx + 1
    if not rows:
        raise ValueError("no drivable roads found in the requested area")
    seg = pd.DataFrame(rows)
    geoms = []
    for pts in seg.pop("_ll"):
        a = np.asarray(pts)
        x, y = proj.to_xy(a[:, 0], a[:, 1])
        geoms.append(LineString(np.column_stack([x, y])))
    seg = gpd.GeoDataFrame(seg, geometry=geoms, crs=proj.crs)
    seg["length_m"] = seg.length.round(1)

    degree = Counter(seg["from_node"]) + Counter(seg["to_node"])
    uf = _UnionFind()
    for a, b, ln, rb in zip(seg["from_node"], seg["to_node"], seg["length_m"], seg["roundabout"]):
        if rb:
            uf.union(a, b)
        elif ln < cluster_m and degree[a] >= 3 and degree[b] >= 3:
            uf.union(a, b)
    in_roundabout = set(seg.loc[seg["roundabout"], "from_node"]) | set(seg.loc[seg["roundabout"], "to_node"])
    members = defaultdict(list)
    for n in degree:
        if degree[n] >= 3 or n in in_roundabout:
            members[uf.find(n)].append(n)
    node_cluster = {n: root for root, ns in members.items() for n in ns}

    fc = seg["from_node"].map(node_cluster)
    tc = seg["to_node"].map(node_cluster)
    seg["junction_internal"] = fc.notna() & (fc == tc) & (seg["roundabout"] | (seg["length_m"] < cluster_m))
    seg["from_intersection"] = fc.map(lambda v: f"X{int(v)}" if pd.notna(v) else None)
    seg["to_intersection"] = tc.map(lambda v: f"X{int(v)}" if pd.notna(v) else None)
    seg["from_degree"] = seg["from_node"].map(degree)
    seg["to_degree"] = seg["to_node"].map(degree)

    # keep segments whose midpoint lies in the study area (degrees were computed before clipping)
    mid = seg.geometry.interpolate(0.5, normalized=True)
    mlon, mlat = proj.to_lonlat(mid.x.values, mid.y.values)
    inside = bbox.contains(mlon, mlat)
    seg["mid_lon"], seg["mid_lat"] = np.round(mlon, 6), np.round(mlat, 6)
    seg = seg[inside].reset_index(drop=True)

    # intersections
    arms, names, internal = defaultdict(int), defaultdict(set), defaultdict(list)
    for r in seg.itertuples():
        if r.junction_internal:
            internal[r.from_intersection].append(r.segment_id)
            if r.roundabout and isinstance(r.name, str):
                names[r.from_intersection].add(r.name)
            continue
        for x in (r.from_intersection, r.to_intersection):
            if x:
                arms[x] += 1
                if isinstance(r.name, str):
                    names[x].add(r.name)
    sig = [(e["lon"], e["lat"]) for e in elements if e["type"] == "node"
           and e.get("tags", {}).get("highway") == "traffic_signals"]
    crs_nodes = [(e["lon"], e["lat"]) for e in elements if e["type"] == "node"
                 and (e.get("tags", {}).get("highway") == "crossing" or "crossing" in e.get("tags", {}))]

    def tree(pts):
        if not pts:
            return None
        a = np.asarray(pts)
        return cKDTree(np.column_stack(proj.to_xy(a[:, 0], a[:, 1])))

    sig_t, crs_t = tree(sig), tree(crs_nodes)
    out = []
    for root, ns in members.items():
        xid = f"X{int(root)}"
        if xid not in arms and xid not in internal:
            continue  # entirely outside the study area
        ll = np.asarray([node_ll[n] for n in ns])
        x, y = proj.to_xy(ll[:, 0], ll[:, 1])
        cx, cy = float(np.mean(x)), float(np.mean(y))
        span = float(np.hypot(x - cx, y - cy).max()) * 2 if len(ns) > 1 else 0.0
        r = span / 2 + 25.0
        lon, lat = proj.to_lonlat(cx, cy)
        out.append({
            "intersection_id": xid,
            "kind": "roundabout" if any(n in in_roundabout for n in ns) else "junction",
            "lon": round(lon, 6), "lat": round(lat, 6),
            "n_arms": arms.get(xid, 0),
            "n_osm_nodes": len(ns),
            "osm_node_ids": ";".join(str(n) for n in sorted(ns)),
            "street_names": "; ".join(sorted(names.get(xid, []))),
            "n_streets": len(names.get(xid, [])),
            "traffic_signals": bool(sig_t is not None and sig_t.query_ball_point([cx, cy], r)),
            "n_crossings": len(crs_t.query_ball_point([cx, cy], r)) if crs_t is not None else 0,
            "node_span_m": round(span, 1),
            "geometry": Point(cx, cy),
        })
    inter = gpd.GeoDataFrame(out, geometry="geometry", crs=proj.crs)
    if len(inter):
        inter = inter[inter["n_arms"] >= 1].reset_index(drop=True)
    _mark_dual_carriageways(seg)
    log.info("network: %d segments (%d junction-internal), %d intersections",
             len(seg), int(seg["junction_internal"].sum()), len(inter))
    return seg, inter


def _bearing(line: LineString) -> float:
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360.0)


def _mark_dual_carriageways(seg: gpd.GeoDataFrame, max_dist: float = 45.0) -> None:
    """Flag one-way segments that have an antiparallel one-way twin with the same name.

    Sets `carriageways` (1 or 2) and `twin_segment_id`.
    """
    seg["carriageways"] = 1
    seg["twin_segment_id"] = None
    cand = seg[(seg["oneway"] != "both") & ~seg["roundabout"] & ~seg["junction_internal"]
               & seg["name"].notna() & ~seg["highway"].str.endswith("_link")]
    if cand.empty:
        return
    mids = cand.geometry.interpolate(0.5, normalized=True)
    xy = np.column_stack([mids.x.values, mids.y.values])
    bearing = np.array([(_bearing(g) + (180 if o == "backward" else 0)) % 360
                        for g, o in zip(cand.geometry, cand["oneway"])])
    tree = cKDTree(xy)
    names, ids, idx = cand["name"].values, cand["segment_id"].values, cand.index.values
    geoms = cand.geometry.values
    for i, nbrs in enumerate(tree.query_ball_point(xy, max_dist)):
        best, best_d = None, 1e9
        for j in nbrs:
            if j == i or names[j] != names[i]:
                continue
            diff = abs((bearing[i] - bearing[j] + 180) % 360 - 180)
            if diff < 150:
                continue
            d = geoms[j].distance(Point(xy[i]))
            if 2.0 < d < best_d:
                best, best_d = j, d
        if best is not None:
            seg.at[idx[i], "carriageways"] = 2
            seg.at[idx[i], "twin_segment_id"] = ids[best]
