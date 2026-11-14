"""One Drex agent per piece: each checks its own spot, unhappy ones move.

After a layout is built, every piece's agent sees a close-up of the page
around itself (its real neighbours, readable text) and Drex answers: does
this piece fit here on all its edges? Pieces that do not fit negotiate:
Drex compares close-ups of the piece in its current spot, in free spots
next to the page, and swapped with another unhappy piece, and picks one.
After each move every agent checks again, so neighbours react to moves.
"""

from concurrent.futures import ThreadPoolExecutor

from .arrange import _gain
import cv2

from . import fit
from .build import STEPS, place
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url

CONTEXT = ("Part of a torn printed page being put back together, shown on a "
           "dark background. The fragment of interest is in the middle of "
           "each image.")
FITS = ("Does the fragment in the middle of Image 1 fit its neighbours: do "
        "text lines, words and printed shapes continue correctly across every "
        "edge where it touches another fragment?")
MOVE = ("Images 1 to {n} show the same fragment, in the middle, placed in "
        "different spots of the page. In which image does it fit its "
        "neighbours best, with text and shapes continuing across its edges?")


def _view(up: dict, placed: dict, focus: str, maxdim: int = 720):
    """Close-up of the page around one piece, with its real neighbours."""
    img = fit.compose(up, placed)
    x0 = min(v[0] for v in placed.values())
    y0 = min(v[1] for v in placed.values())
    fx, fy = placed[focus][0] - x0, placed[focus][1] - y0
    h, w = up[focus].shape[:2]
    m = int(0.6 * max(h, w))
    img = img[max(0, fy - m):fy + h + m, max(0, fx - m):fx + w + m]
    s = maxdim / max(img.shape[:2])
    return cv2.resize(img, None, fx=min(1, s), fy=min(1, s),
                      interpolation=cv2.INTER_AREA)


def _fits(up, placed, pieces, ask_fn, model, workers):
    def one(p):
        res = ask_fn(CONTEXT, {"fits": {"type": "noul", "instructions": FITS}},
                     model=model, images=[data_url(_view(up, placed, p), 720)])
        return p, float(res.body["answers"]["fits"]["noul"]), res
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, pieces))
    return {p: v for p, v, _ in out}, [r for _, _, r in out]


def _pick(images, ask_fn, model):
    crit = {f"image_{i + 1}": f"Image {i + 1}." for i in range(len(images))}
    res = ask_fn(CONTEXT, {"spot": {"type": "choice", "criteria": crit,
                                    "instructions": MOVE.format(
                                        n=len(images))}},
                 model=model, images=[data_url(i, 720) for i in images])
    probs = res.body["answers"]["spot"]["probabilities"]
    return int(max(probs, key=probs.get)[6:]) - 1, probs, res


def repair(up: dict, cells: dict, best: dict, right: dict, below: dict,
           base, rounds: int = 3, ask_fn=default_ask,
           model: str = VISION_MODEL, workers: int = 16, on_step=None):
    """Piece agents check and fix the layout. Returns (cells, placed, calls)."""
    calls = []
    placed = place(up, cells, best)
    for _ in range(rounds):
        happy, c = _fits(up, placed, list(cells), ask_fn, model, workers)
        calls += c
        med = sorted(happy.values())[len(happy) // 2]
        # Drex's fit scores are low for everyone; only a clear outlier moves
        unhappy = sorted((p for p in happy if happy[p] < 0.6 * med),
                         key=happy.get)
        if on_step:
            on_step({"happy": {p: round(v, 2) for p, v in happy.items()}})
        if not unhappy:
            break
        moved = False
        for w in unhappy[:1]:
            at = {v: k for k, v in cells.items()}
            rest = {k: v for k, v in at.items() if v != w}
            free = {(r + dr, c + dc) for r, c in rest
                    for dr, dc in STEPS} - set(rest)
            spots = sorted(free, key=lambda s: -_gain(
                w, s, cells, rest, right, below, base)[0])[:5]
            options = [cells] + [{**cells, w: s} for s in spots]
            options += [{**cells, w: cells[u], u: cells[w]}
                        for u in unhappy if u != w][:2]
            with ThreadPoolExecutor(max_workers=8) as pool:
                places = list(pool.map(lambda o: place(up, o, best), options))
            k, probs, res = _pick([_view(up, pl, w) for pl in places],
                                  ask_fn, model)
            calls.append(res)
            if on_step:
                on_step({"piece": w, "options": len(options), "pick": k,
                         "p": round(probs[f"image_{k + 1}"], 2)})
            if k and probs[f"image_{k + 1}"] >= probs["image_1"] + 0.15:
                cells, placed, moved = options[k], places[k], True
        if not moved:
            break
    return cells, placed, calls


REVIEW = ("Image 1 is a torn page put back together; each fragment carries a "
          "red number. Is the page correctly rebuilt: every fragment in its "
          "right place and turned the right way, text reading across tears?")
WORST = ("In Image 1, which numbered fragment is most clearly in the wrong "
         "place or turned the wrong way? Answer none if all are right.")


def _numbered(up, placed):
    img = fit.compose(up, placed).copy()
    x0 = min(v[0] for v in placed.values())
    y0 = min(v[1] for v in placed.values())
    names = sorted(placed)
    size = max(2.0, img.shape[1] / 500)
    for n, p in enumerate(names, 1):
        h, w = up[p].shape[:2]
        c = (placed[p][0] - x0 + w // 2, placed[p][1] - y0 + h // 2)
        cv2.putText(img, str(n), c, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (40, 40, 230), int(size * 3), cv2.LINE_AA)
    return img, names


def review(up: dict, cells: dict, placed: dict, best: dict, right: dict,
           below: dict, base, rounds: int = 3, ask_fn=default_ask,
           model: str = VISION_MODEL, on_step=None):
    """Drex reviews the whole page; the piece it flags is re-placed (spots,
    swaps, other turns) from close-ups, and tray pieces get a chance too.
    Returns (cells, placed, calls)."""
    from .arrange import base as stem
    calls = []
    for _ in range(rounds):
        img, names = _numbered(up, placed)
        crit = {f"piece_{n}": f"Fragment {n}." for n in range(1, len(names) + 1)}
        crit["none"] = "None: every fragment is right."
        res = ask_fn(CONTEXT, {"ok": {"type": "noul", "instructions": REVIEW},
                               "worst": {"type": "choice", "instructions": WORST,
                                         "criteria": crit}},
                     model=model, images=[data_url(img, 1000)])
        calls.append(res)
        ok = float(res.body["answers"]["ok"]["noul"])
        probs = res.body["answers"]["worst"]["probabilities"]
        worst = max(probs, key=probs.get)
        if on_step:
            on_step({"review_ok": round(ok, 2), "flagged": worst})
        used = {stem(p) for p in cells}
        movers = [] if worst == "none" or ok >= 0.6 else \
            [names[int(worst[6:]) - 1]]
        movers += [p for p in up if stem(p) not in used
                   and p.split("@")[-1] in (p, "0")][:2]  # tray pieces
        if not movers:
            break
        for w in movers:
            at = {v: k for k, v in cells.items()}
            rest = {k: v for k, v in at.items() if v != w}
            free = {(r + dr, c + dc) for r, c in rest
                    for dr, dc in STEPS} - set(rest)
            spots = sorted(free, key=lambda s: -_gain(
                w, s, cells, rest, right, below, base)[0])[:4]
            twins = [t for t in up if stem(t) == stem(w) and t != w]
            options = ([cells] if w in cells else []) + \
                [{**{k: v for k, v in cells.items() if k != w}, w: s}
                 for s in spots]
            if w in cells:
                options += [{**{k: v for k, v in cells.items() if k != w},
                             t: cells[w]} for t in twins]
                options += [{**cells, w: cells[u], u: cells[w]}
                            for u in sorted(cells, key=lambda u: -_gain(
                                w, cells[u], cells, rest, right, below,
                                base)[0]) if u != w][:2]
            options = options[:8]
            if len(options) < 2 and (w in cells or not options):
                continue  # nothing to choose between
            with ThreadPoolExecutor(max_workers=8) as pool:
                places = list(pool.map(lambda o: place(up, o, best), options))
            focus = [w if w in o else next(t for t in twins if t in o)
                     for o in options]
            k, pr, res2 = _pick([_view(up, pl, f) for pl, f in
                                 zip(places, focus)], ask_fn, model)
            calls.append(res2)
            keep_first = w in cells
            better = pr[f"image_{k + 1}"] >= (pr["image_1"] + 0.1
                                              if keep_first else 0.3)
            if (k or not keep_first) and better:
                cells, placed = options[k], places[k]
            if on_step:
                on_step({"piece": w, "options": len(options), "pick": k,
                         "p": round(pr[f"image_{k + 1}"], 2)})
    return cells, placed, calls
