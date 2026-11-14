"""Pick the piece finder that fits the photo's background.

A plain, even background (a studio or rendered scene, like ShredBench's
grey) needs nothing clever: paper is brighter than it, so Otsu's
threshold separates them, and touching pieces stay apart because the
shadows between them are darker still. A real table with texture,
sunlight and objects at the edges goes to photo.find_pieces.
"""

import cv2
import numpy as np

from .photo import MIN_SHARE, find_pieces

PLAIN = 8.0  # lightness spread along the photo's edge below which it is plain


def is_plain(image) -> bool:
    s = 1280 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    b = max(4, int(0.04 * min(L.shape)))
    ring = np.concatenate([L[:b].ravel(), L[-b:].ravel(),
                           L[:, :b].ravel(), L[:, -b:].ravel()])
    return float(ring.std()) < PLAIN


def _fill(mask):
    h, w = mask.shape
    ff = np.pad(mask, 1).copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return mask | (ff[1:-1, 1:-1] == 0).astype(np.uint8)


def bright_pieces(image) -> list:
    """Masks of the pieces that are brighter than a plain background."""
    L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    L = cv2.GaussianBlur(L, (0, 0), 2)
    _, m = cv2.threshold(L, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = _fill(cv2.morphologyEx(m, cv2.MORPH_OPEN, k))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    floor = max(MIN_SHARE, 0.003) * m.size
    return [(lab == i).astype(np.uint8) for i in range(1, n)
            if st[i][4] >= floor]


def find(image) -> list:
    """Candidate piece masks, full size, with the finder that fits."""
    return bright_pieces(image) if is_plain(image) else find_pieces(image)
