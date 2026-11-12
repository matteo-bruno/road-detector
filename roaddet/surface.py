"""Road surface detection: a per-pixel classifier trained with weak labels from OSM.

Positive samples are pixels on the road centre lines, negatives are pixels inside building
footprints and pixels far from any road. No manual annotation is needed, so the model adapts
to each city and imagery source.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

log = logging.getLogger(__name__)

FEATURES = ["L", "a", "b", "sat", "exg", "L_mean", "a_mean", "b_mean", "L_std_s", "L_std_l", "grad"]


def vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((2 * g - r - b) > 22) & (g > r)


def pixel_features(rgb: np.ndarray, m_per_px: float) -> np.ndarray:
    """Return float32 features (H, W, F) for an RGB uint8 image."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    f32 = rgb.astype(np.float32)
    exg = 2 * f32[..., 1] - f32[..., 0] - f32[..., 2]
    ks = max(3, int(round(1.5 / m_per_px)) | 1)   # ~1.5 m window
    kl = max(5, int(round(4.0 / m_per_px)) | 1)   # ~4 m window
    mean = cv2.blur(lab, (ks, ks))
    L = lab[..., 0]

    def local_std(k):
        m = cv2.blur(L, (k, k))
        return np.sqrt(np.maximum(cv2.blur(L * L, (k, k)) - m * m, 0))

    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.blur(np.hypot(gx, gy), (ks, ks))
    return np.dstack([lab[..., 0], lab[..., 1], lab[..., 2], hsv[..., 1].astype(np.float32), exg,
                      mean[..., 0], mean[..., 1], mean[..., 2], local_std(ks), local_std(kl), grad])


def rasterize_lines(shape, lines_px: list[np.ndarray], thickness: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.polylines(mask, [np.round(l).astype(np.int32) for l in lines_px], False, 1, thickness=max(1, thickness))
    return mask


def rasterize_polygons(shape, polys_px: list[list[np.ndarray]]) -> np.ndarray:
    """polys_px: list of [outer, hole, hole, ...] rings in pixel coordinates."""
    mask = np.zeros(shape, np.uint8)
    for rings in polys_px:
        cv2.fillPoly(mask, [np.round(rings[0]).astype(np.int32)], 1)
    for rings in polys_px:
        for hole in rings[1:]:
            cv2.fillPoly(mask, [np.round(hole).astype(np.int32)], 0)
    return mask


def weak_labels(rgb, road_lines_px, all_way_lines_px, bld_mask, m_per_px):
    """Return label image: 1 road, 0 non-road, 255 unlabeled."""
    h, w = rgb.shape[:2]
    core = rasterize_lines((h, w), road_lines_px, int(round(1.6 / m_per_px)))
    near = rasterize_lines((h, w), all_way_lines_px, int(round(30.0 / m_per_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 / m_per_px) | 1,) * 2)
    bld_core = cv2.erode(bld_mask, k)
    veg = vegetation_mask(rgb)
    lab = np.full((h, w), 255, np.uint8)
    lab[(near == 0) & (bld_mask == 0)] = 0   # courtyards, gardens, parks
    lab[bld_core == 1] = 0                   # roofs
    lab[(core == 1) & (bld_mask == 0) & ~veg] = 1
    lab[rgb.sum(axis=2) == 0] = 255          # missing imagery
    return lab


def sample_training(feat, lab, n_per_class=150_000, rng=None):
    rng = rng or np.random.default_rng(0)
    X, y = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(lab.ravel() == cls)
        if len(idx) == 0:
            X.append(np.zeros((0, feat.shape[2]), np.float32))
            y.append(np.zeros(0, np.uint8))
            continue
        idx = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        X.append(feat.reshape(-1, feat.shape[2])[idx])
        y.append(np.full(len(idx), cls, np.uint8))
    return np.concatenate(X), np.concatenate(y)


def train(X, y) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.15,
                                         class_weight="balanced", random_state=0)
    n = len(y)
    perm = np.random.default_rng(1).permutation(n)
    cut = int(n * 0.85)
    clf.fit(X[perm[:cut]], y[perm[:cut]])
    acc = float((clf.predict(X[perm[cut:]]) == y[perm[cut:]]).mean())
    log.info("surface classifier: %d samples, hold-out accuracy on weak labels %.3f", n, acc)
    clf.weak_label_accuracy_ = acc
    return clf


def predict(clf, rgb, m_per_px, mask=None, block=1536, pad=24) -> np.ndarray:
    """Road probability as uint8 (0..255). Only pixels where mask is set are evaluated."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), np.uint8)
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            r1, c1 = min(h, r0 + block), min(w, c0 + block)
            sel = slice(r0, r1), slice(c0, c1)
            m = np.ones((r1 - r0, c1 - c0), bool) if mask is None else mask[sel].astype(bool)
            if not m.any():
                continue
            ra, ca = max(0, r0 - pad), max(0, c0 - pad)
            feat = pixel_features(rgb[ra:min(h, r1 + pad), ca:min(w, c1 + pad)], m_per_px)
            feat = feat[r0 - ra:r0 - ra + (r1 - r0), c0 - ca:c0 - ca + (c1 - c0)]
            p = clf.predict_proba(feat[m])[:, 1]
            blk = np.zeros(m.shape, np.uint8)
            blk[m] = np.round(p * 255).astype(np.uint8)
            out[sel] = blk
    return out
