"""Matcher: piece agents ask Drex which pieces are their neighbors.

Each piece is an agent with a local view. For every piece A one Drex
request asks all of A's right-candidate and below-candidate questions
plus "is A on the right edge?". Output: score tables 0..1 and per-call
metrics. The answer key is never part of any request.
"""

from dataclasses import dataclass, field

from .drex import DEFAULT_MODEL, CallResult, ask as default_ask
from .shredder import Piece, Shred


@dataclass
class PieceAgent:
    piece: Piece
    right: dict[str, float] = field(default_factory=dict)
    below: dict[str, float] = field(default_factory=dict)
    right_edge: float = 0.0

    @property
    def id(self) -> str:
        return self.piece.piece_id

    @property
    def text(self) -> str:
        return "\n".join(self.piece.lines)


@dataclass
class MatchResult:
    agents: dict[str, PieceAgent]
    calls: list[CallResult]

    @property
    def right(self) -> dict[str, dict[str, float]]:
        return {a.id: dict(a.right) for a in self.agents.values()}

    @property
    def below(self) -> dict[str, dict[str, float]]:
        return {a.id: dict(a.below) for a in self.agents.values()}

    @property
    def right_edge(self) -> dict[str, float]:
        return {a.id: a.right_edge for a in self.agents.values()}


def _join_right(a: Piece, b: Piece) -> str:
    n = max(len(a.lines), len(b.lines))
    out = []
    for i in range(n):
        la = a.lines[i] if i < len(a.lines) else ""
        lb = b.lines[i] if i < len(b.lines) else ""
        out.append(la + lb)
    return "\n".join(out)


def _stack(a: Piece, b: Piece) -> str:
    return "\n".join(list(a.lines) + list(b.lines))


def _build_request(agent: PieceAgent, others: list[PieceAgent]) -> dict:
    cand_right = {o.id: _join_right(agent.piece, o.piece) for o in others}
    cand_below = {o.id: _stack(agent.piece, o.piece) for o in others}
    state = {
        "piece_A": agent.text,
        "candidates_right": cand_right,
        "candidates_below": cand_below,
    }
    questions = {
        "right_edge": {
            "type": "noul",
            "instructions": (
                "Looking at piece_A alone: is piece_A on the right edge of "
                "the page, i.e. no page text follows it to the right?"
            ),
        }
    }
    for o in others:
        questions[f"right_{o.id}"] = {
            "type": "noul",
            "instructions": (
                f"candidates_right['{o.id}'] is piece_A joined line by line "
                f"with candidate '{o.id}'. Is that joined text one "
                "continuous natural passage, with words and line breaks "
                "intact and nothing missing or duplicated?"
            ),
        }
        questions[f"below_{o.id}"] = {
            "type": "noul",
            "instructions": (
                f"candidates_below['{o.id}'] is piece_A stacked directly "
                f"above candidate '{o.id}'. Is that stacked text one "
                "continuous natural passage, with words and line breaks "
                "intact and nothing missing or duplicated?"
            ),
        }
    return {"state": state, "questions": questions}


def match(shred: Shred, model: str = DEFAULT_MODEL,
          ask_fn=default_ask, **ask_kwargs) -> MatchResult:
    """Run neighbor matching for every piece. One Drex call per piece."""
    agents = {p.piece_id: PieceAgent(piece=p) for p in shred.pieces}
    calls: list[CallResult] = []
    for agent in agents.values():
        others = [a for a in agents.values() if a.id != agent.id]
        req = _build_request(agent, others)
        res = ask_fn(req["state"], req["questions"], model=model,
                     **ask_kwargs)
        calls.append(res)
        answers = res.body["answers"]
        agent.right_edge = answers["right_edge"]["noul"]
        for o in others:
            agent.right[o.id] = answers[f"right_{o.id}"]["noul"]
            agent.below[o.id] = answers[f"below_{o.id}"]["noul"]
    return MatchResult(agents=agents, calls=calls)


def neighbor_stats(result: MatchResult,
                   answer_key: dict[str, tuple[int, int]],
                   rows: int, cols: int) -> tuple[int, int]:
    """(correct, total) over right- and below-neighbor picks."""
    total = correct = 0
    pos_of = {pid: pos for pid, pos in answer_key.items()}
    id_at = {pos: pid for pid, pos in answer_key.items()}
    for pid, (r, c) in pos_of.items():
        total += 1  # right neighbor (or right edge)
        if c + 1 < cols:
            true_id = id_at[(r, c + 1)]
            pred = max((b for b in result.agents[pid].right),
                       key=lambda b: result.agents[pid].right[b])
            if pred == true_id:
                correct += 1
        elif result.agents[pid].right_edge >= 0.5:
            correct += 1
        if r + 1 < rows:  # below neighbor
            total += 1
            true_id = id_at[(r + 1, c)]
            pred = max((b for b in result.agents[pid].below),
                       key=lambda b: result.agents[pid].below[b])
            if pred == true_id:
                correct += 1
    return correct, total


def neighbor_accuracy(result: MatchResult,
                      answer_key: dict[str, tuple[int, int]],
                      rows: int, cols: int) -> float:
    """Share of neighbor picks (right or below) that match the true key."""
    correct, total = neighbor_stats(result, answer_key, rows, cols)
    return correct / total
