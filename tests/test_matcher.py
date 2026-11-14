"""Tests for unshred.matcher (no network: Drex is replaced by a fake)."""
import json

import pytest

from unshred.drex import CallResult
from unshred.matcher import (
    _join_right,
    _stack,
    match,
    neighbor_accuracy,
    neighbor_stats,
)
from unshred.shredder import Piece, shred

TEXT = " ".join(f"word{i:03d}" for i in range(60))
ROWS, COLS = 2, 3


class FakeDrex:
    """Stands in for drex.ask. Scores from the key; records every request."""

    def __init__(self, shr, hit=0.9, miss=0.1):
        self.key = shr.answer_key
        self.id_at = {pos: pid for pid, pos in shr.answer_key.items()}
        self.cols = shr.cols
        self.hit, self.miss = hit, miss
        self.requests = []

    def __call__(self, state, questions, **kwargs):
        self.requests.append(
            {"state": state, "questions": questions, "kwargs": kwargs})
        (a_id,) = set(self.key) - set(state["candidates_right"])
        r, c = self.key[a_id]
        answers = {}
        for qid in questions:
            if qid == "right_edge":
                yes = c == self.cols - 1
            else:
                direction, b_id = qid.split("_", 1)
                want = (r, c + 1) if direction == "right" else (r + 1, c)
                yes = self.id_at.get(want) == b_id
            answers[qid] = {"type": "noul",
                            "noul": self.hit if yes else self.miss}
        body = {"answers": answers, "usage": {"input_tokens": 100}}
        return CallResult(body=body, latency_ms=12.5, input_tokens=100,
                          request_id=f"r{len(self.requests)}",
                          model=kwargs.get("model", ""))


@pytest.fixture
def shr():
    return shred(TEXT, rows=ROWS, cols=COLS, seed=1)


def test_join_right_and_stack():
    a = Piece("a", ("ab", "cd"))
    b = Piece("b", ("12", "34"))
    assert _join_right(a, b) == "ab12\ncd34"
    assert _stack(a, b) == "ab\ncd\n12\n34"


def test_join_right_pads_missing_lines():
    a = Piece("a", ("ab", "cd"))
    b = Piece("b", ("12",))
    assert _join_right(a, b) == "ab12\ncd"


def test_one_call_per_piece_with_metrics(shr):
    fake = FakeDrex(shr)
    res = match(shr, ask_fn=fake)
    n = ROWS * COLS
    assert len(fake.requests) == n
    assert len(res.calls) == n
    assert all(c.latency_ms == 12.5 and c.input_tokens == 100
               for c in res.calls)


def test_score_tables_cover_every_other_piece(shr):
    res = match(shr, ask_fn=FakeDrex(shr))
    ids = set(shr.answer_key)
    assert set(res.right) == set(res.below) == set(res.right_edge) == ids
    for a in ids:
        assert set(res.right[a]) == ids - {a}
        assert set(res.below[a]) == ids - {a}
        scores = list(res.right[a].values()) + list(res.below[a].values())
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_scores_are_read_from_drex_answers(shr):
    res = match(shr, ask_fn=FakeDrex(shr))
    id_at = {pos: pid for pid, pos in shr.answer_key.items()}
    a, right_b, below_b = id_at[(0, 0)], id_at[(0, 1)], id_at[(1, 0)]
    assert res.right[a][right_b] == 0.9
    assert res.below[a][below_b] == 0.9
    assert res.right[a][below_b] == 0.1
    assert res.right_edge[a] == 0.1
    assert res.right_edge[id_at[(0, COLS - 1)]] == 0.9


def test_request_shape_and_questions(shr):
    fake = FakeDrex(shr)
    match(shr, ask_fn=fake)
    by_id = {p.piece_id: p for p in shr.pieces}
    for req in fake.requests:
        state, questions = req["state"], req["questions"]
        assert set(state) == {"piece_A", "candidates_right",
                              "candidates_below"}
        others = set(state["candidates_right"])
        assert others == set(state["candidates_below"])
        (a_id,) = set(by_id) - others
        assert state["piece_A"] == "\n".join(by_id[a_id].lines)
        expected = {"right_edge"}
        expected |= {f"right_{o}" for o in others}
        expected |= {f"below_{o}" for o in others}
        assert set(questions) == expected
        assert all(q["type"] == "noul" and q["instructions"]
                   for q in questions.values())
        for o in others:
            assert state["candidates_right"][o] == _join_right(
                by_id[a_id], by_id[o])
            assert state["candidates_below"][o] == _stack(
                by_id[a_id], by_id[o])


def test_answer_key_is_never_sent(shr):
    fake = FakeDrex(shr)
    match(shr, ask_fn=fake)
    for req in fake.requests:
        blob = json.dumps({"s": req["state"], "q": req["questions"]})
        assert "answer_key" not in blob
        for value in req["state"].values():
            # only text goes out: no (row, col) positions anywhere
            texts = value.values() if isinstance(value, dict) else [value]
            assert all(isinstance(t, str) for t in texts)


def test_model_and_kwargs_are_passed_through(shr):
    fake = FakeDrex(shr)
    res = match(shr, model="drex-test", ask_fn=fake, timeout=5.0)
    assert all(r["kwargs"] == {"model": "drex-test", "timeout": 5.0}
               for r in fake.requests)
    assert res.calls[0].model == "drex-test"


def test_neighbor_stats_perfect(shr):
    res = match(shr, ask_fn=FakeDrex(shr))
    # one right pick (or right-edge call) per piece + one below pick per
    # piece that is not in the last row
    total = ROWS * COLS + (ROWS - 1) * COLS
    assert neighbor_stats(res, shr.answer_key, ROWS, COLS) == (total, total)
    assert neighbor_accuracy(res, shr.answer_key, ROWS, COLS) == 1.0


def test_neighbor_stats_all_wrong(shr):
    res = match(shr, ask_fn=FakeDrex(shr, hit=0.1, miss=0.9))
    total = ROWS * COLS + (ROWS - 1) * COLS
    assert neighbor_stats(res, shr.answer_key, ROWS, COLS) == (0, total)
    assert neighbor_accuracy(res, shr.answer_key, ROWS, COLS) == 0.0
