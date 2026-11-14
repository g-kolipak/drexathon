"""Rebuild a torn page from a photo, with Drex Max checking pieces together.

  uv run --env-file .env scripts/rebuild_photo.py image.png
  uv run scripts/rebuild_photo.py image.png --reuse     re-lay out only

1. OpenCV cuts the pieces out and straightens them (no model).
2. Drex Max turns each piece upright (one request per piece).
3. For every ordered pair and both directions, the pieces are pushed
   together at their 3 best text-line alignments and drawn as ONE image.
   Drex Max gives its confidence that each joined image is a true match;
   a pair's score is its best alignment's confidence.
4. arrange.py checks pieces together: a layout scores the evidence of
   EVERY pair of neighbours in it (log-odds of Drex's confidence), and a
   beam search keeps the best layouts. Pieces with no positive evidence
   go to a "needs human" list with their top candidates.

Output in runs/photo/NAME/: rebuilt.jpg, joins/ (the accepted links as
Drex saw them) and joined_scores.json (every score, for --reuse).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import arrange, cutouts, fit, scoring, tear  # noqa: E402
from unshred.drex import VISION_MODEL, ask  # noqa: E402
from unshred.photo import find_pieces  # noqa: E402

CONTEXT = ("Torn fragments of one printed page. Each image shows two "
           "fragments pushed together at their torn edges, on a dark "
           "background.")
ASK = {
    "right": ("Image {i}: the second fragment sits directly to the right of "
              "the first. Is this how they fit on the original page: does "
              "every line of text run straight across the join, with words "
              "and letters continuing correctly, and do printed shapes line "
              "up?"),
    "below": ("Image {i}: the second fragment sits directly below the first, "
              "in the same column. Is this how they fit on the original "
              "page: do the margins, boxes and lines of print continue "
              "straight down across the join, as one piece of the page?"),
}


def score_pairs(up: dict, k: int, workers: int):
    """Every ordered pair, both directions, k alignments: Drex confidence."""
    jobs = [(a, b, d, f) for d in ("right", "below") for a in up for b in up
            if a != b for f in fit.fits(up[a], up[b], d, k)]

    def one(start):
        batch = jobs[start:start + 8]  # Drex Max reads up to 8 images
        images = [tear.data_url(fit.joined(up[a], up[b], f, 640))
                  for a, b, _, f in batch]
        questions = {f"join_{n + 1}": {
            "type": "noul", "instructions": ASK[d].format(i=n + 1)}
            for n, (_, _, d, _) in enumerate(batch)}
        res = ask(CONTEXT, questions, model=VISION_MODEL, images=images)
        probs = [float(res.body["answers"][f"join_{n + 1}"]["noul"])
                 for n in range(len(batch))]
        return list(zip(batch, probs)), res

    best, calls = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pairs, res in pool.map(one, range(0, len(jobs), 8)):
            calls.append(res)
            for (a, b, d, f), p in pairs:
                key = f"{a}|{b}|{d}"
                if key not in best or p > best[key]["p"]:
                    best[key] = {"p": round(p, 4), "dx": f[0], "dy": f[1]}
    return best, calls


def _places(up: dict, cells: dict, links: list, best: dict, gap: int = 60):
    """Pixel places: start from one piece and walk the neighbour pairs,
    each placed at the alignment Drex picked; pieces with several placed
    neighbours take the average. Unplaced pieces go in a row below."""
    if not cells:
        return fit.layout(up, [], {})
    first = min(cells, key=lambda p: cells[p])
    places = {first: (0.0, 0.0)}
    while True:
        votes = {}
        for a, b, d, _ in links:
            v = best[f"{a}|{b}|{d}"]
            if a in places and b not in places:
                votes.setdefault(b, []).append((places[a][0] + v["dx"],
                                                places[a][1] + v["dy"]))
            if b in places and a not in places:
                votes.setdefault(a, []).append((places[b][0] - v["dx"],
                                                places[b][1] - v["dy"]))
        if not votes:
            break
        for p, vs in votes.items():
            places[p] = (sum(x for x, _ in vs) / len(vs),
                         sum(y for _, y in vs) / len(vs))
    places = {p: (int(x), int(y)) for p, (x, y) in places.items()}
    y = max(places[p][1] + up[p].shape[0] for p in places) + gap
    x = min(v[0] for v in places.values())
    for p in up:
        if p not in places:
            places[p] = (x, y)
            x += up[p].shape[1] + gap
    return places


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    image = cv2.imread(args.photo)
    out = ROOT / "runs" / "photo" / Path(args.photo).stem
    (out / "joins").mkdir(parents=True, exist_ok=True)
    saved = out / "joined_scores.json"
    pieces = {f"p{i + 1}": cutouts.straighten(cutouts.cutout(image, m))
              for i, m in enumerate(find_pieces(image))}
    t0 = time.monotonic()
    if args.reuse:
        data = json.loads(saved.read_text())
        turns, best = data["turns"], data["best"]
        up = {p: tear.quarter_turn(pieces[p], turns[p]["degrees"])
              for p in pieces}
    else:
        up, turns, calls = tear.orient_all(pieces, workers=args.workers)
        best, more = score_pairs(up, args.k, args.workers)
        use = scoring.usage(calls + more)
        saved.write_text(json.dumps({"turns": turns, "best": best,
                                     "usage": use}, indent=1))
        print(f"Drex Max: {use['calls']} calls ({len(pieces)} upright, "
              f"{len(more)} joined checks), {use['input_tokens']} input "
              f"tokens, wall {time.monotonic() - t0:.1f} s")
    right = {p: {} for p in up}
    below = {p: {} for p in up}
    for key, v in best.items():
        a, b, d = key.split("|")
        (right if d == "right" else below)[a][b] = v["p"]
    cells, score, links = arrange.arrange(right, below, beam=args.beam)
    rows = 1 + max(r for r, _ in cells.values())
    cols = 1 + max(c for _, c in cells.values())
    at = {v: k for k, v in cells.items()}
    print(f"Layout checked together (score {score:.1f}, {len(cells)} of "
          f"{len(up)} pieces):")
    for r in range(rows):
        print("   " + " ".join(f"{at.get((r, c), '..'):>3}" for c in range(cols)))
    print("Every neighbour pair in it, with Drex's confidence:")
    for a, b, d, p in sorted(links, key=lambda x: -x[3]):
        note = "" if p >= 0.5 else "   (Drex doubts this one)"
        print(f"  {b} {d:5s} of {a}  {p:.2f}{note}")
        v = best[f"{a}|{b}|{d}"]
        img = fit.joined(up[a], up[b], (v["dx"], v["dy"]), 900)
        cv2.imwrite(str(out / "joins" / f"{a}_{d}_{b}.jpg"), img)
    for item in arrange.unplaced(cells, right, below):
        cands = ", ".join(f"{c['link']} {c['score']:.2f}"
                          for c in item["candidates"])
        print(f"  needs human: {item['piece']}  (best: {cands})")
    places = _places(up, cells, links, best)
    cv2.imwrite(str(out / "rebuilt.jpg"), fit.compose(up, places))
    print(f"Saved to {out}/ (rebuilt.jpg, joins/, joined_scores.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
