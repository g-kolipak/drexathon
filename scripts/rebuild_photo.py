"""Rebuild a torn page from a photo. Every decision is Drex Max's.

  uv run --env-file .env scripts/rebuild_photo.py image.png
  uv run --env-file .env scripts/rebuild_photo.py image.png --fast
  uv run scripts/rebuild_photo.py image.png --reuse     re-lay out only

1. find      OpenCV proposes every region that might be a piece, traces
             its outline (in parallel threads) and wins back paper bent
             into shadow (shade.py).
2. check     Drex, one call per region, all in parallel: is it paper, and
             which quarter turn is upright.
3. rate      OpenCV pre-screens pairs (text-line rhythm, colour, torn
             edge fit) and draws joined images at 3 alignments; Drex rates
             every one yes/no, all in parallel. --fast skips this and
             trusts the pre-screen (about 9 s faster, less accurate).
4. compare   Drex, in parallel: for each piece and side, which of its top
             candidates is the true neighbour (pick one, or none).
5. page      the best complete layouts from Drex's picks are drawn as
             full pages with the tears closed; Drex picks the right page,
             then re-decides the least supported pieces.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import arrange, build, compare, cutouts, fit, plain, scoring, screen, shade  # noqa: E402,E501
from unshred.drex import VISION_MODEL, ask  # noqa: E402
from unshred.photo import find_pieces  # noqa: E402

CONTEXT = ("Torn fragments of one printed page. Each image shows two "
           "fragments pushed together at their torn edges, on a dark "
           "background.")
ASK = {"right": "Image {i}: the second fragment sits directly to the right "
                "of the first. Does every line of text and every printed "
                "shape continue correctly across the join?",
       "below": "Image {i}: the second fragment sits directly below the "
                "first, in the same column. Do the margins, boxes and lines "
                "of print continue straight down across the join?"}


def rate_pairs(up: dict, workers: int):
    """Drex rates every shortlisted pair and alignment, in parallel."""
    jobs = screen.shortlist(up, k=3)

    def one(start):
        batch = jobs[start:start + 8]  # Drex Max reads up to 8 images
        res = ask(CONTEXT, {f"join_{n + 1}": {
            "type": "noul", "instructions": ASK[d].format(i=n + 1)}
            for n, (_, _, d, _) in enumerate(batch)}, model=VISION_MODEL,
            images=[fit_url(up, a, b, f) for a, b, _, f in batch])
        return [(job, float(res.body["answers"][f"join_{n + 1}"]["noul"]))
                for n, job in enumerate(batch)], res

    best, calls = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pairs, res in pool.map(one, range(0, len(jobs), 8)):
            calls.append(res)
            for (a, b, d, f), p in pairs:
                key = f"{a}|{b}|{d}"
                if key not in best or p > best[key]["p"]:
                    best[key] = {"p": round(p, 4), "dx": f[0], "dy": f[1]}
    return best, calls


def fit_url(up, a, b, f):
    from unshred.tear import data_url
    return data_url(fit.joined(up[a], up[b], f, 640))


class Clock:
    def __init__(self):
        self.t0 = self.t = time.monotonic()

    def __call__(self, label: str, calls=None):
        now = time.monotonic()
        extra = f"  ({len(calls)} Drex calls)" if calls is not None else ""
        print(f"  {now - self.t:6.2f} s  {label}{extra}")
        self.t = now


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    clock = Clock()
    image = cv2.imread(args.photo)
    out = ROOT / "runs" / "photo" / Path(args.photo).stem
    (out / "joins").mkdir(parents=True, exist_ok=True)
    saved = out / "scores.json"
    masks = (plain.find(image) if plain.is_plain(image)
             else shade.grow(image, find_pieces(image)))
    pieces = {f"p{i + 1}": cutouts.straighten(cutouts.cutout(image, m))
              for i, m in enumerate(masks)}
    where = {f"p{i + 1}": [float(v) for v in np.argwhere(m > 0).mean(0)[::-1]]
             for i, m in enumerate(masks)}
    clock(f"find: {len(masks)} candidate regions (OpenCV)")
    if args.reuse:
        data = json.loads(saved.read_text())
    else:
        data, calls = {}, []
        paper, turns, upright, c1 = screen.check(pieces, workers=args.workers)
        calls += c1
        clock("check: paper or not + which way up", c1)
        up = {p: upright[p] for p in pieces if paper[p] >= 0.5}
        if not args.fast:
            best, c2 = rate_pairs(up, args.workers)
            calls += c2
            clock("screen + Drex rates every shortlisted pair", c2)
        else:
            best = screen.shortlist(up, as_best=True)
            clock("screen: shortlist pairs and alignments (OpenCV)")
        right, below, c3 = compare.compare(up, best, workers=args.workers)
        calls += c3
        clock("compare: Drex picks each piece's true neighbours", c3)
        data = {"paper": paper, "turns": turns, "best": best,
                "pick": {"right": right, "below": below},
                "usage": scoring.usage(calls)}
        saved.write_text(json.dumps(data, indent=1))
    paper, turns, best = data["paper"], data["turns"], data["best"]
    up = {p: np.ascontiguousarray(np.rot90(pieces[p], k=(-turns[p]["degrees"]
                                                        // 90) % 4))
          for p in pieces if paper[p] >= 0.5}
    right, below = data["pick"]["right"], data["pick"]["below"]
    if not up:
        print("  Drex found no paper pieces; nothing to rebuild")
        return 1
    dropped = [f"{p} ({paper[p]:.2f})" for p in pieces if p not in up]
    print(f"  Drex: {len(up)} of {len(pieces)} regions are paper"
          + (f"; not paper: {', '.join(dropped)}" if dropped else ""))
    layouts = arrange.arrange(right, below, base=1 / 8, top=8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        places = list(pool.map(lambda lay: build.place(up, lay[0], best),
                               layouts))
    pick, probs, res = build.choose([fit.compose(up, pl) for pl in places])
    choice = pick if pick is not None else 0
    clock(f"page: Drex picked layout {choice + 1} of {len(layouts)}", [res])
    cells, placed, c4 = build.improve(
        up, layouts[choice][0], best, right, below, 1 / 8, rounds=2,
        on_round=lambda w, n, k, pr: print(
            f"           Drex re-checked {w} in {n} spots: "
            + ("kept it" if not k else f"moved it ({pr[f'image_{k + 1}']:.2f})")))
    clock("page: Drex re-decided the weakest pieces", c4)
    cells = arrange._normal(cells)
    at = {v: k for k, v in cells.items()}
    for r in range(1 + max(r for r, _ in cells.values())):
        print("     " + " ".join(f"{at.get((r, c), '..'):>3}" for c in
                                 range(1 + max(c for _, c in cells.values()))))
    left = [p for p in up if p not in placed]
    page = fit.compose(up, placed)
    if left:  # leftover pieces in a tray under the page, for a human
        tray = fit.compose(up, fit.layout({p: up[p] for p in left}, [], {}))
        w = max(page.shape[1], tray.shape[1])
        page, tray = (cv2.copyMakeBorder(i, top, 0, 0, w - i.shape[1],
                                         cv2.BORDER_CONSTANT, value=fit.BACK)
                      for i, top in ((page, 0), (tray, 80)))
        page = np.vstack([page, tray])
        print(f"  needs human: {', '.join(left)}")
    cv2.imwrite(str(out / "rebuilt.jpg"), page)
    (out / "layout.json").write_text(json.dumps({
        "cells": cells, "placed": placed, "links": arrange._links(
            cells, right, below), "where": {p: where[p] for p in up}},
        indent=1))
    print(f"  total {time.monotonic() - clock.t0:.1f} s -> {out}/rebuilt.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
