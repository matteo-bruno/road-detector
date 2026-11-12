"""Cross-section measurement of street segments on the imagery.

Each segment is "straightened": the image is resampled on a grid whose rows follow the OSM
centre line (axis s) and whose columns are perpendicular offsets (axis t, positive = left of
the digitised direction). On this strip every element of the street (travel lanes, rows of
parked cars, kerbs, facades) is a band parallel to the s axis, which makes them measurable
with simple, inspectable profile statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import shapely

from .surface import vegetation_mask


@dataclass
class StripParams:
    half_width_m: float = 22.0
    ds: float = 0.5          # along-axis sampling step (m)
    dt: float = 0.25         # cross-axis sampling step (m)
    block_m: float = 20.0    # length of the blocks measured independently
    end_trim_m: float = 10.0  # ignored length next to an intersection
    min_usable_m: float = 8.0
    centre_search_m: float = 5.0
    road_thr: float = 0.5
    max_parking_m: float = 6.5
    min_parking_m: float = 1.6
    max_sidewalk_m: float = 8.0
    min_sidewalk_m: float = 0.8


def strip_maps(line, proj, grid, p: StripParams):
    """Pixel-coordinate maps (float32) of the straightened strip of a metric LineString."""
    n = max(2, int(line.length / p.ds) + 1)
    s = np.linspace(0.0, line.length, n)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
    k = max(1, int(round(2.0 / p.ds)))
    a = pts[np.clip(np.arange(n) + k, 0, n - 1)] - pts[np.clip(np.arange(n) - k, 0, n - 1)]
    norm = np.hypot(a[:, 0], a[:, 1])
    norm[norm == 0] = 1.0
    tx, ty = a[:, 0] / norm, a[:, 1] / norm
    nx, ny = -ty, tx  # left normal
    lon0, lat0 = proj.to_lonlat(pts[:, 0], pts[:, 1])
    lon1, lat1 = proj.to_lonlat(pts[:, 0] + nx, pts[:, 1] + ny)
    c0, r0 = grid.lonlat_to_px(lon0, lat0)
    c1, r1 = grid.lonlat_to_px(lon1, lat1)
    t = np.arange(-p.half_width_m, p.half_width_m + 1e-6, p.dt)
    mx = (c0[:, None] + t[None, :] * (c1 - c0)[:, None]).astype(np.float32)
    my = (r0[:, None] + t[None, :] * (r1 - r0)[:, None]).astype(np.float32)
    return mx, my, s, t


def sample(arr: np.ndarray, mx, my, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Sample arr at the map positions, reading only the needed window."""
    h, w = arr.shape[:2]
    c0, c1 = int(max(0, np.floor(mx.min()) - 2)), int(min(w, np.ceil(mx.max()) + 3))
    r0, r1 = int(max(0, np.floor(my.min()) - 2)), int(min(h, np.ceil(my.max()) + 3))
    if c1 <= c0 or r1 <= r0:
        return np.zeros(mx.shape + arr.shape[2:], arr.dtype)
    return cv2.remap(arr[r0:r1, c0:c1], mx - c0, my - r0, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _smooth(v, k):
    return v if k <= 1 else np.convolve(v, np.ones(k) / k, mode="same")


def _run_end(ok: np.ndarray, start: int, step: int, max_gap: int) -> int:
    """Walk from start in direction step while ok (allowing short gaps); return last ok index."""
    i, last, gap = start, start, 0
    n = len(ok)
    while 0 <= i + step < n:
        i += step
        if ok[i]:
            last, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last


def _block_measure(P, lum, veg, bld, t, p: StripParams, c0: int | None):
    """Measure one block. Returns dict or None if the travel way is not visible."""
    dt = p.dt
    a = _smooth(np.median(P, axis=0), 3)
    q = _smooth((P > p.road_thr).mean(axis=0), 3)
    w2 = int(round(2.0 / dt))
    win = np.convolve(a, np.ones(w2) / w2, mode="same")
    mid = len(t) // 2
    sr = int(round(p.centre_search_m / dt))
    lo, hi = mid - sr, mid + sr + 1
    if c0 is None:
        # prefer positions near the OSM axis: small distance penalty
        score = win[lo:hi] - 0.02 * np.abs(t[lo:hi])
        c0 = lo + int(np.argmax(score))
    if win[c0] < 0.45:
        return None
    ok = a >= p.road_thr
    if not ok[c0]:
        near = np.flatnonzero(ok[max(0, c0 - 4):c0 + 5])
        if len(near) == 0:
            return None
        c0 = max(0, c0 - 4) + int(near[np.argmin(np.abs(near - 4))])
    gap = int(round(0.5 / dt))
    iL, iR = _run_end(ok, c0, -1, gap), _run_end(ok, c0, +1, gap)
    # t grows with the index: lower index = right side, higher index = left side
    res = {"t_right": t[iL] - dt / 2, "t_left": t[iR] + dt / 2}
    res["travel_w"] = res["t_left"] - res["t_right"]
    if iL <= 1 or iR >= len(t) - 2:
        res["open"] = True  # band reaches the strip limit: square, car park or misclassification
    # Parked cars sit inside the detected surface band, along its margins. A column belongs to
    # a parking row when many of its pixels differ from the asphalt brightness (occupancy) and
    # brightness varies strongly along the axis (alternating cars and gaps, unlike a shadow).
    cen = slice(max(iL, c0 - int(1 / dt)), min(iR, c0 + int(1 / dt)) + 1)
    ref = float(np.median(lum[:, cen]))
    occ = _smooth((np.abs(lum - ref) > 22).mean(axis=0), 3)
    sigma = _smooth(lum.std(axis=0), 3)
    base = float(np.percentile(sigma[iL:iR + 1], 25))
    vegf = veg.mean(axis=0)
    bldf = bld.mean(axis=0) if bld is not None else np.zeros_like(a)
    car = (occ > 0.3) & (sigma > max(1.5 * base, base + 6.0)) & (vegf < 0.35) & (bldf < 0.5)
    band = iR - iL + 1
    nmax = int(round(p.max_parking_m / dt))
    lead = int(round(1.0 / dt))

    def run(start, step, limit):
        """Length (columns) of the car-like run starting at start, walking by step."""
        j, n, gap, length = start, 0, 0, 0
        while n < limit and 0 <= j < len(t):
            if car[j]:
                gap, length = 0, n + 1
            else:
                gap += 1
                if gap > 1:
                    break
            j += step
            n += 1
        return length

    edges = {}
    for side, edge, step in (("right", iL, +1), ("left", iR, -1)):  # step points inwards
        # a row of parked cars can be classified as surface (inside the band) or not (outside)
        n_out = run(edge - step, -step, nmax)
        j, skipped = edge, 0
        while skipped < lead and not car[j] and iL < j < iR:
            j += step
            skipped += 1
        n_in = run(j, step, min(nmax - n_out, int(0.45 * band)))
        n_in = n_in + skipped if n_in else 0
        pw = (n_out + n_in) * dt
        if pw < p.min_parking_m:
            pw, n_out, n_in = 0.0, 0, 0
        edge = int(np.clip(edge - step * n_out, 0, len(t) - 1))  # kerb: outer limit of the parking row
        edges[side] = edge
        res[f"park_{side}"] = pw
        lo_, hi_ = sorted((edge, edge + step * max(0, n_out + n_in - 1)))
        res[f"occ_{side}"] = float(occ[lo_:hi_ + 1].mean()) if pw else np.nan
        # outside the band: canopy next to the edge hides the kerb; first building pixel = facade
        out_step = -step
        o1 = edge + out_step
        o2 = edge + out_step * int(round(2.0 / dt))
        lo_, hi_ = max(0, min(o1, o2)), min(len(t) - 1, max(o1, o2))
        res[f"canopy_edge_{side}"] = float(vegf[lo_:hi_ + 1].mean()) if hi_ >= lo_ else 0.0
        sw = np.nan
        if bld is not None:
            rng = range(edge + out_step, len(t) if out_step > 0 else -1, out_step)
            hit = next((jj for jj in rng if bldf[jj] >= 0.5), None)
            if hit is not None:
                d = (abs(hit - edge) - 1) * dt
                between = vegf[min(edge, hit) + 1:max(edge, hit)]
                if p.min_sidewalk_m <= d <= p.max_sidewalk_m and (between.mean() if len(between) else 0) < 0.5:
                    sw = d
                res[f"facade_{side}"] = abs(t[hit])
        res[f"sidewalk_{side}"] = sw
    res["t_right"], res["t_left"] = t[edges["right"]] - dt / 2, t[edges["left"]] + dt / 2
    res["travel_w"] = res["t_left"] - res["t_right"]
    res["c0"] = c0
    return res


def measure_strip(rgb, prob, bld, s, t, p: StripParams, trim_start: float, trim_end: float) -> dict:
    """Aggregate block measurements for one segment strip."""
    out = {"measured": False, "n_blocks": 0}
    L = s[-1]
    rows = np.flatnonzero((s >= trim_start) & (s <= L - trim_end))
    if len(rows) * p.ds < p.min_usable_m:
        out["fail_reason"] = "too_short"
        return out
    rgb, prob = rgb[rows], prob[rows]
    bld = bld[rows] if bld is not None else None
    valid = rgb.sum(axis=2) > 0
    if valid.mean() < 0.7:
        out["fail_reason"] = "no_imagery"
        return out
    P = prob.astype(np.float32) / 255.0
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    veg = vegetation_mask(rgb)
    mid = len(t) // 2
    cband = slice(mid - int(3 / p.dt), mid + int(3 / p.dt) + 1)
    out["canopy_frac"] = round(float(veg[:, cband].mean()), 3)
    out["shadow_frac"] = round(float((lum[:, cband] < 45).mean()), 3)
    out["road_prob_axis"] = round(float(P[:, cband].mean()), 3)

    # global pass fixes the lateral position of the travel way, blocks refine it
    g = _block_measure(P, lum, veg, bld, t, p, None)
    if g is None:
        out["fail_reason"] = "canopy" if out["canopy_frac"] > 0.4 else "not_visible"
        return out
    nb = max(1, int(round(len(rows) * p.ds / p.block_m)))
    blocks = []
    for idx in np.array_split(np.arange(len(rows)), nb):
        if len(idx) < 6:
            continue
        b = _block_measure(P[idx], lum[idx], veg[idx], None if bld is None else bld[idx], t, p, g["c0"])
        if b is not None and not b.get("open"):
            blocks.append(b)
    out["n_blocks"] = nb
    if len(blocks) < max(1, 0.3 * nb):
        out["fail_reason"] = "open_area" if g.get("open") else "inconsistent"
        return out

    def col(k):
        return np.array([b.get(k, np.nan) for b in blocks], dtype=float)

    cw = col("travel_w")
    out["measured"] = True
    out["carriageway_width_m"] = float(np.median(cw))
    out["carriageway_width_iqr_m"] = float(np.subtract(*np.percentile(cw, [75, 25])))
    out["axis_offset_m"] = float(np.median((col("t_left") + col("t_right")) / 2))
    for side in ("left", "right"):
        pk = col(f"park_{side}")
        share = float((pk > 0).mean())
        has = share >= 0.5
        out[f"parking_{side}_share"] = round(share, 2)
        out[f"parking_{side}_width_m"] = float(np.median(pk[pk > 0])) if has else 0.0
        out[f"parking_{side}_occupancy"] = float(np.nanmean(col(f"occ_{side}"))) if has else np.nan
        sw = col(f"sidewalk_{side}")
        out[f"sidewalk_{side}_width_m"] = float(np.nanmedian(sw)) if np.isfinite(sw).mean() >= 0.5 else np.nan
        fc = col(f"facade_{side}")
        out[f"facade_{side}_m"] = float(np.nanmedian(fc)) if np.isfinite(fc).mean() >= 0.5 else np.nan
        out[f"canopy_edge_{side}"] = float(np.nanmean(col(f"canopy_edge_{side}")))
    out["block_valid_frac"] = round(len(blocks) / nb, 2)
    return out


def debug_strip(rgb, prob, bld, s, t, res: dict, p: StripParams) -> np.ndarray:
    """Side-by-side RGB strip and probability strip with the measured edges drawn."""
    scale = 3
    h, w = rgb.shape[:2]
    a = rgb.copy()
    pr = cv2.applyColorMap(prob, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    if bld is not None:
        pr[bld > 0] = (pr[bld > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    def xcol(tv):
        return int(round((tv - t[0]) / p.dt))

    for im in (a, pr):
        im[:, xcol(0)] = (255, 255, 0)
        if res.get("measured"):
            off, half = res["axis_offset_m"], res["carriageway_width_m"] / 2
            for tv, colr in ((off - half, (0, 255, 255)), (off + half, (0, 255, 255)),
                             (off - half + res["parking_right_width_m"], (255, 0, 255)),
                             (off + half - res["parking_left_width_m"], (255, 0, 255))):
                x = xcol(tv)
                if 0 <= x < w:
                    im[:, x] = colr
    both = np.hstack([a, np.full((h, 4, 3), 255, np.uint8), pr])
    return cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
