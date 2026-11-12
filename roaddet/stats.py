"""City-level statistics, comparison with OpenStreetMap tags, and a Markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_ORDER = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
               "residential", "living_street", "pedestrian", "service"]


def describe(v: pd.Series, w: pd.Series | None = None) -> dict:
    """Summary statistics of v; w (segment length) gives a length-weighted mean and median."""
    ok = v.notna()
    v = v[ok].astype(float)
    if len(v) == 0:
        return {"n": 0}
    out = {"n": int(len(v)), "mean": v.mean(), "median": v.median(), "std": v.std(ddof=0),
           "min": v.min(), "p10": v.quantile(0.10), "p25": v.quantile(0.25),
           "p75": v.quantile(0.75), "p90": v.quantile(0.90), "max": v.max()}
    if w is not None:
        ww = w[ok].astype(float).values
        order = np.argsort(v.values)
        cum = np.cumsum(ww[order]) / ww.sum()
        out["mean_length_weighted"] = float(np.average(v.values, weights=ww))
        out["median_length_weighted"] = float(v.values[order][np.searchsorted(cum, 0.5)])
    return {k: (round(float(x), 2) if k != "n" else x) for k, x in out.items()}


def _class(hw: pd.Series) -> pd.Series:
    return hw.str.replace("_link", "", regex=False)


def compute(seg: pd.DataFrame, inter: pd.DataFrame, meta: dict) -> dict:
    seg = seg[~seg["junction_internal"]].copy()
    seg["cls"] = _class(seg["highway"])
    m = seg[seg["measured"]]
    good = m[m["quality"].isin(["good", "fair"])]
    L = seg["length_m"]
    km = L.sum() / 1000
    st = {"meta": meta}
    st["network"] = {
        "segments": int(len(seg)), "length_km": round(km, 1),
        "streets_named": int(seg["name"].nunique()),
        "measured_segments": int(len(m)), "measured_share_of_length": round(m["length_m"].sum() / L.sum(), 3),
        "good_or_fair_share_of_length": round(good["length_m"].sum() / L.sum(), 3),
        "not_measured_reasons": seg.loc[~seg["measured"], "fail_reason"].value_counts().to_dict(),
        "quality_counts": seg["quality"].value_counts().to_dict(),
        "one_way_share_of_length": round(L[seg["oneway"] != "both"].sum() / L.sum(), 3),
        "dual_carriageway_share_of_length": round(L[seg["carriageways"] == 2].sum() / L.sum(), 3),
        "segment_length_m": describe(L),
    }
    area = meta.get("area_km2") or np.nan
    ra = good["road_area_m2"].sum()
    # extrapolate paved area to unmeasured segments with the mean width of their class
    cls_w = good.groupby("cls")["carriageway_width_m"].mean()
    rest = seg[~seg.index.isin(good.index)]
    ra_all = ra + (rest["length_m"] * rest["cls"].map(cls_w).fillna(good["carriageway_width_m"].mean())).sum()
    st["road_area"] = {"measured_km2": round(ra / 1e6, 3), "estimated_total_km2": round(ra_all / 1e6, 3),
                       "share_of_study_area": round(ra_all / 1e6 / area, 3) if area else None}

    def widths(df):
        return {
            "carriageway_width_m": describe(df["carriageway_width_m"], df["length_m"]),
            "travel_width_m": describe(df["travel_width_m"], df["length_m"]),
            "building_to_building_m": describe(df["building_to_building_m"], df["length_m"]),
        }

    st["widths_all_measured"] = widths(m)
    st["widths_good_fair"] = widths(good)
    st["carriageway_by_class"] = {c: describe(g["carriageway_width_m"], g["length_m"])
                                  for c, g in sorted(good.groupby("cls"), key=lambda kv: CLASS_ORDER.index(kv[0])
                                                     if kv[0] in CLASS_ORDER else 99)}
    st["carriageway_by_osm_lanes"] = {str(int(k)): describe(g["carriageway_width_m"], g["length_m"])
                                      for k, g in good.dropna(subset=["lanes_osm"]).groupby("lanes_osm")}
    st["carriageway_by_direction"] = {k: describe(g["carriageway_width_m"], g["length_m"])
                                      for k, g in good.assign(d=np.where(good["oneway"] == "both", "two_way", "one_way")).groupby("d")}

    pw = pd.concat([good["parking_left_width_m"], good["parking_right_width_m"]])
    occ = pd.concat([good["parking_left_occupancy"], good["parking_right_occupancy"]])
    types = pd.concat([good["parking_left_type"], good["parking_right_type"]]).value_counts(normalize=True)
    gl = good["length_m"]
    st["parking"] = {
        "share_of_length_by_sides": {str(int(k)): round(gl[good["parking_sides"] == k].sum() / gl.sum(), 3) for k in (0, 1, 2)},
        "width_m": describe(pw[pw > 0]),
        "occupancy": describe(occ),
        "type_share_of_sides": {k: round(float(v), 3) for k, v in types.items()},
        "kerb_length_with_parking_km": round(float((good["length_m"] * good["parking_sides"]).sum() / 1000), 1),
    }
    sw = pd.concat([good["sidewalk_left_width_m"], good["sidewalk_right_width_m"]])
    st["sidewalks"] = {
        "share_of_length_by_sides": {str(int(k)): round(gl[good["sidewalk_sides"] == k].sum() / gl.sum(), 3) for k in (0, 1, 2)},
        "width_m": describe(sw),
    }
    ms = seg.dropna(subset=["maxspeed_kmh"])
    st["speed_limits_share_of_length"] = {
        str(int(k)): round(v / ms["length_m"].sum(), 3) for k, v in ms.groupby("maxspeed_kmh")["length_m"].sum().items()}
    st["speed_limit_source_share"] = seg.groupby("maxspeed_source")["length_m"].sum().div(L.sum()).round(3).to_dict()
    st["lanes_osm_share_of_length"] = {
        str(int(k)): round(v / seg.dropna(subset=["lanes_osm"])["length_m"].sum(), 3)
        for k, v in seg.groupby("lanes_osm")["length_m"].sum().items()}

    # comparison with OSM tags
    cmp = {}
    c = good.dropna(subset=["lanes_osm", "lanes_est"])
    if len(c):
        d = c["lanes_est"].astype(float) - c["lanes_osm"]
        cmp["lanes"] = {"n": int(len(c)), "exact": round(float((d == 0).mean()), 3),
                        "within_1": round(float((d.abs() <= 1).mean()), 3), "mean_diff": round(float(d.mean()), 2)}
    c = good.dropna(subset=["width_osm_m"])
    if len(c):
        d = c["carriageway_width_m"] - c["width_osm_m"]
        cmp["width_tag"] = {"n": int(len(c)), "mae_m": round(float(d.abs().mean()), 2),
                            "bias_m": round(float(d.mean()), 2), "median_abs_m": round(float(d.abs().median()), 2)}
    rows = []
    for side in ("left", "right"):
        t = good[[f"parking_osm_{side}", f"parking_{side}_width_m"]].dropna()
        rows.append(pd.DataFrame({"osm": t.iloc[:, 0] != "no", "det": t.iloc[:, 1] > 0}))
    t = pd.concat(rows)
    if len(t):
        cmp["parking_presence"] = {"n_sides": int(len(t)), "agreement": round(float((t["osm"] == t["det"]).mean()), 3),
                                   "osm_yes_detected": round(float(t.loc[t["osm"], "det"].mean()), 3) if t["osm"].any() else None,
                                   "osm_no_detected": round(float(t.loc[~t["osm"], "det"].mean()), 3) if (~t["osm"]).any() else None}
    t = good[good["sidewalk_osm"].isin(["both", "left", "right", "no"])]
    if len(t):
        n_osm = t["sidewalk_osm"].map({"both": 2, "left": 1, "right": 1, "no": 0})
        cmp["sidewalk_sides"] = {"n": int(len(t)), "exact": round(float((n_osm == t["sidewalk_sides"]).mean()), 3)}
    st["osm_comparison"] = cmp

    x = inter
    st["intersections"] = {
        "count": int(len(x)), "per_km2": round(len(x) / area, 1) if area else None,
        "per_km_of_road": round(len(x) / km, 2),
        "by_arms": {str(int(k)): int(v) for k, v in x["n_arms"].value_counts().sort_index().items()},
        "roundabouts": int((x["kind"] == "roundabout").sum()),
        "with_traffic_signals": int(x["traffic_signals"].sum()),
        "approx_diameter_m": describe(x["approx_diameter_m"]),
        "paved_area_m2": describe(x["paved_area_m2"]),
    }
    top = good[good["name"].notna()].groupby("name").agg(
        length_m=("length_m", "sum"), w=("carriageway_width_m", "median"), n=("segment_id", "count"))
    top = top[top["length_m"] >= 200]
    st["widest_streets"] = top.sort_values("w", ascending=False).head(10).round(1).reset_index().to_dict("records")
    st["narrowest_streets"] = top.sort_values("w").head(10).round(1).reset_index().to_dict("records")
    return st


def _table(d: dict, key_name: str, keys=("n", "mean", "mean_length_weighted", "median", "std", "p10", "p25", "p75", "p90")) -> str:
    head = "| " + key_name + " | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n"
    return head + "".join("| " + str(k) + " | " + " | ".join(str(v.get(c, "")) for c in keys) + " |\n"
                          for k, v in d.items() if v.get("n"))


def report(st: dict, plots: list[str]) -> str:
    m, n = st["meta"], st["network"]
    w = st["widths_good_fair"]
    lines = [
        f"# Road statistics: {m.get('label') or 'study area'}", "",
        f"Area {m['area_km2']} km2, bbox {m['bbox']}. Imagery: {m['imagery']} "
        f"({m['ground_resolution_m']} m/pixel). Street network: {m['osm']}. Generated {m['created']}.", "",
        "## Network", "",
        f"- {n['segments']} street segments, {n['length_km']} km, {n['streets_named']} named streets",
        f"- measured on imagery: {n['measured_share_of_length']:.0%} of the length "
        f"({n['good_or_fair_share_of_length']:.0%} with good or fair quality)",
        f"- not measured: {n['not_measured_reasons']}",
        f"- one-way: {n['one_way_share_of_length']:.0%} of the length, dual carriageways: "
        f"{n['dual_carriageway_share_of_length']:.0%}",
        f"- paved carriageway area: {st['road_area']['estimated_total_km2']} km2 "
        f"({st['road_area']['share_of_study_area']:.1%} of the study area)", "",
        "## Widths (metres, segments with good or fair quality)", "",
        _table({"carriageway (kerb to kerb, includes parking rows)": w["carriageway_width_m"],
                "travel way (carriageway minus parking rows)": w["travel_width_m"],
                "building to building": w["building_to_building_m"],
                "parking row": st["parking"]["width_m"],
                "sidewalk (kerb to facade)": st["sidewalks"]["width_m"]}, "quantity"),
        "", "### Carriageway width by road class", "", _table(st["carriageway_by_class"], "class"),
        "", "### Carriageway width by OSM lane count", "", _table(st["carriageway_by_osm_lanes"], "lanes"),
        "", "### Carriageway width by direction", "", _table(st["carriageway_by_direction"], "direction"),
        "", "## Parking and sidewalks", "",
        f"- share of street length with parking rows on 0 / 1 / 2 sides: {st['parking']['share_of_length_by_sides']}",
        f"- kerb length with parking: {st['parking']['kerb_length_with_parking_km']} km; "
        f"mean occupancy of detected rows: {st['parking']['occupancy'].get('mean')}",
        f"- row type (share of kerb sides): {st['parking']['type_share_of_sides']}",
        f"- share of street length with a measurable sidewalk on 0 / 1 / 2 sides: {st['sidewalks']['share_of_length_by_sides']}",
        "", "## Speed limits, lanes (from OSM)", "",
        f"- speed limit share of length (km/h): {st['speed_limits_share_of_length']}",
        f"- speed limit source: {st['speed_limit_source_share']}",
        f"- lanes share of length: {st['lanes_osm_share_of_length']}",
        "", "## Agreement with OSM tags", "",
    ]
    for k, v in st["osm_comparison"].items():
        lines.append(f"- {k}: {v}")
    x = st["intersections"]
    lines += ["", "## Intersections", "",
              f"- {x['count']} intersections ({x['per_km2']} per km2, {x['per_km_of_road']} per km of road), "
              f"{x['roundabouts']} roundabouts, {x['with_traffic_signals']} with traffic signals",
              f"- by number of arms: {x['by_arms']}",
              f"- approximate diameter (m): {x['approx_diameter_m']}",
              "", "## Widest and narrowest streets (median carriageway, at least 200 m measured)", "",
              "| widest | m | narrowest | m |", "|---|---|---|---|"]
    for a, b in zip(st["widest_streets"], st["narrowest_streets"]):
        lines.append(f"| {a['name']} | {a['w']} | {b['name']} | {b['w']} |")
    lines.append("")
    for p in plots:
        lines += [f"![{p}]({p})", ""]
    return "\n".join(lines)


def plots(seg, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = seg[seg["measured"] & seg["quality"].isin(["good", "fair"]) & ~seg["junction_internal"]].copy()
    good["cls"] = _class(good["highway"])
    names = []
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 36, 1)
    ax.hist(good["carriageway_width_m"], bins=bins, weights=good["length_m"] / 1000, color="#3b6ea5")
    med = good["carriageway_width_m"].median()
    ax.axvline(med, color="#222", lw=1)
    ax.text(med + 0.4, ax.get_ylim()[1] * 0.93, f"median {med:.1f} m")
    ax.set_xlabel("carriageway width (m)")
    ax.set_ylabel("street length (km)")
    ax.set_title("Carriageway width distribution", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "width_hist.png", dpi=130)
    plt.close(fig)
    names.append("width_hist.png")

    order = [c for c in CLASS_ORDER if (good["cls"] == c).sum() >= 5]
    if order:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.boxplot([good.loc[good["cls"] == c, "carriageway_width_m"] for c in order], vert=False,
                   showfliers=False, patch_artist=True, boxprops={"facecolor": "#a9c4de"},
                   medianprops={"color": "#222"})
        ax.set_yticks(range(1, len(order) + 1), order)
        ax.invert_yaxis()
        ax.set_xlabel("carriageway width (m)")
        ax.set_title("Carriageway width by road class", loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / "width_by_class.png", dpi=130)
        plt.close(fig)
        names.append("width_by_class.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    seg[~seg["measured"]].plot(ax=ax, color="#cccccc", linewidth=0.5)
    good.sort_values("carriageway_width_m").plot(ax=ax, column="carriageway_width_m", cmap="viridis_r", vmin=4, vmax=24,
                                                linewidth=1.3, legend=True,
                                                legend_kwds={"label": "carriageway width (m)", "shrink": 0.6})
    ax.set_aspect(1 / np.cos(np.radians(seg["mid_lat"].mean())))
    ax.set_axis_off()
    ax.set_title("Measured carriageway width (grey: not measured)", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "width_map.png", dpi=130)
    plt.close(fig)
    names.append("width_map.png")
    return names


def _plain(o):
    """Convert numpy scalars to plain Python numbers, recursively."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def write(seg, inter, meta, out_dir: Path) -> dict:
    st = _plain(compute(seg, inter, meta))
    (out_dir / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False, default=float))
    (out_dir / "REPORT.md").write_text(report(st, plots(seg, out_dir)))
    return st
