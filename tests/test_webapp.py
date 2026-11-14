"""Tests for unshred.webapp (no network, no pipeline: subprocess is faked)."""
import time

from fastapi.testclient import TestClient

from unshred import webapp


class FakeProc:
    def __init__(self, cmd, **kw):
        src = webapp.Path(cmd[-1])
        out = webapp.ROOT / "runs" / "photo" / src.stem
        out.mkdir(parents=True, exist_ok=True)
        (out / "rebuilt.jpg").write_bytes(b"\xff\xd8fake")
        self.stdout = iter(["  1.00 s  find: 2 candidate regions\n",
                            "  total 1.0 s\n"])

    def wait(self):
        return 0


def test_upload_runs_the_pipeline_and_serves_the_result(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "ROOT", tmp_path)
    monkeypatch.setattr(webapp, "UPLOADS", tmp_path / "runs" / "uploads")
    monkeypatch.setattr(webapp.subprocess, "Popen", FakeProc)
    client = TestClient(webapp.app)
    jid = client.post("/api/rebuild", content=b"jpegbytes").json()["id"]
    for _ in range(50):
        job = client.get(f"/api/rebuild/{jid}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "done"
    assert any("find" in line for line in job["log"])
    img = client.get(f"/api/rebuild/{jid}/image")
    assert img.status_code == 200 and img.content.startswith(b"\xff\xd8")


def test_bad_requests_are_refused():
    client = TestClient(webapp.app)
    assert client.post("/api/rebuild", content=b"").status_code == 400
    assert client.get("/api/rebuild/nothere123").status_code == 404
    assert client.get("/api/rebuild/../../etc").status_code == 404
