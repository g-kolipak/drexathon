"""Mask helpers shared by the photo pipeline."""

import cv2
import numpy as np


def _el(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def split(mask, radius: int = 10, min_area: int = 2000):
    """Separate pieces that only touch at a corner or a thin neck."""
    core = cv2.erode(mask, _el(2 * radius + 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 4)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
    if len(keep) < 2:
        return [mask]
    seeds = np.isin(labels, keep)
    _, nearest = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_CCOMP)
    parts = []
    for k in keep:
        ids = np.unique(nearest[labels == k])
        part = ((mask > 0) & np.isin(nearest, ids)).astype(np.uint8)
        parts.append(largest(part))
    return parts


def largest(mask):
    """The biggest blob of a mask, with its holes filled."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return out
