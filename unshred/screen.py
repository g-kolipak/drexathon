"""What Drex is asked, so the photo pipeline works for any number of pieces.

  is_paper    Drex Max decides which candidate regions are torn paper
              (not table, shadow, cloth, a device...). This replaces any
              fixed size or position rule for what counts as a piece.
  shortlist   which pairs Drex checks joined together. Up to `all_upto`
              pieces: every pair. Beyond that a cheap pre-screen (text
              line rhythm and colour continuity along the touching edges)
              keeps each piece's best few partners per side, so the
              number of Drex checks grows with n, not n squared.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import fit
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url

PAPER_CONTEXT = ("Regions cut out of a photo of a table on which torn "
                 "paper fragments lie. Each image shows one region on a "
                 "plain dark background.")
PAPER_Q = ("Image {i}: is this a torn fragment of a paper page (printed or "
           "blank paper with torn or cut edges), rather than part of the "
           "table, a shadow, cloth, a hand, a device or any other object?")


def is_paper(pieces: dict, ask_fn=default_ask, model: str = VISION_MODEL,
             workers: int = 4) -> tuple[dict, list]:
    """{piece id: Drex's probability that it is paper}, and the calls."""
    ids = list(pieces)

    def one(start):
        batch = ids[start:start + 8]
        res = ask_fn(PAPER_CONTEXT, {
            f"region_{n + 1}": {"type": "noul",
                                "instructions": PAPER_Q.format(i=n + 1)}
            for n in range(len(batch))}, model=model,
            images=[data_url(pieces[p], 512) for p in batch])
        return {p: float(res.body["answers"][f"region_{n + 1}"]["noul"])
                for n, p in enumerate(batch)}, res

    probs, calls = {}, []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for got, res in pool.map(one, range(0, len(ids), 8)):
            probs.update(got)
            calls.append(res)
    return probs, calls


def _band_colors(piece, side: str, share: float = 0.06):
    """Mean colour per row in a thin band along one side (NaN: no paper)."""
    m = piece[..., 3] > 128
    w = m.shape[1]
    band = max(3, int(share * w))
    right, left = fit._edges(piece)
    cols = np.arange(w)[None, :]
    if side == "right":
        near = (cols <= right[:, None]) & (cols > right[:, None] - band)
    else:
        near = (cols >= left[:, None]) & (cols < left[:, None] + band)
    near &= m
    cnt = near.sum(axis=1, keepdims=True)
    mean = (piece[..., :3] * near[..., None]).sum(axis=1) / np.maximum(cnt, 1)
    return np.where(cnt > 0, mean, np.nan)


def _shape(ra, lb) -> float:
    """How snugly B's torn left edge fills A's torn right edge (higher is
    better): the mean leftover gap at the snuggest vertical offset."""
    ha, hb = len(ra), len(lb)
    need = int(fit.MIN_OVERLAP * min(ha, hb))
    best = 9.0
    for dy in range(-hb + need, ha - need + 1, 2):
        y0, y1 = max(0, dy), min(ha, hb + dy)
        A, B = ra[y0:y1], lb[y0 - dy:y1 - dy]
        ok = (A >= 0) & (B >= 0)
        if ok.sum() >= need:
            g = A[ok] - B[ok]
            best = min(best, float((g.max() - g).mean()) / min(ha, hb))
    return -best


def _quick(a, b, direction: str, k: int):
    """Alignments plus one cheap plausibility number (higher is better):
    text-line rhythm + colour continuity + torn-edge fit + edge length."""
    if direction == "below":
        a, b = fit._t(a), fit._t(b)
    picks = fit.fits_right(a, b, k)
    ca, cb = _band_colors(a, "right"), _band_colors(b, "left")
    dx, dy, corr = picks[0]
    y0, y1 = max(0, dy), min(len(ca), len(cb) + dy)
    diff = np.nanmean(np.abs(ca[y0:y1] - cb[y0 - dy:y1 - dy])) / 255 \
        if y1 > y0 else 1.0
    ra, _ = fit._edges(a)
    _, lb = fit._edges(b)
    la, lb_len = (ra >= 0).sum(), (lb >= 0).sum()
    length = abs(la - lb_len) / max(la, lb_len, 1)
    if direction == "below":
        picks = [(y, x, s) for x, y, s in picks]
    color = 1.0 if np.isnan(diff) else diff
    return picks, corr - 2 * color + 4 * _shape(ra, lb) - length


def shortlist(up: dict, k: int = 3, keep: int = 8, all_upto: int = 12,
              scale: float = 0.5):
    """Jobs (a, b, direction, (dx, dy, score)) worth a Drex check."""
    ids = list(up)
    small = {p: cv2.resize(up[p], None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) for p in ids}
    quick = {}
    for d in ("right", "below"):
        for a in ids:
            for b in ids:
                if a != b:
                    picks, s = _quick(small[a], small[b], d, k)
                    quick[(a, b, d)] = ([(int(x / scale), int(y / scale), c)
                                         for x, y, c in picks], s)
    chosen = set()
    for d in ("right", "below"):
        for p in ids:
            out = sorted((b for b in ids if b != p),
                         key=lambda b: -quick[(p, b, d)][1])
            inn = sorted((a for a in ids if a != p),
                         key=lambda a: -quick[(a, p, d)][1])
            n = len(ids) if len(ids) <= all_upto else keep
            chosen |= {(p, b, d) for b in out[:n]}
            chosen |= {(a, p, d) for a in inn[:n]}
    return [(a, b, d, f) for (a, b, d) in sorted(chosen)
            for f in quick[(a, b, d)][0]]
