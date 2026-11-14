"""Website: upload a photo of torn pieces, get the page Drex rebuilt.

  uv run --env-file .env uvicorn unshred.webapp:app --port 8000
  then open http://127.0.0.1:8000

Each upload runs scripts/rebuild_photo.py (every decision by Drex Max) in
the background; the page polls for its progress lines and then shows the
rebuilt page. The key comes from the server's environment only.
"""

import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "runs" / "uploads"
MAX_BYTES = 40_000_000
JOBS: dict = {}
app = FastAPI(title="Unshred")


def _job(jid: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{10}", jid) or jid not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[jid]


def _run(jid: str, src: Path) -> None:
    """Run the pipeline on one upload, keeping its progress lines."""
    job = JOBS[jid]
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "rebuild_photo.py"), str(src)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        job["log"].append(line.rstrip())
    code = proc.wait()
    job["status"] = "done" if code == 0 and job["image"].exists() else "failed"


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "rebuild.html")


@app.post("/api/rebuild")
async def start(request: Request):
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(400, "send one image file, under 40 MB")
    jid = uuid.uuid4().hex[:10]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    src = UPLOADS / f"up_{jid}.jpg"
    src.write_bytes(body)
    JOBS[jid] = {"status": "running", "log": [], "source": src,
                 "image": ROOT / "runs" / "photo" / src.stem / "rebuilt.jpg"}
    threading.Thread(target=_run, args=(jid, src), daemon=True).start()
    return {"id": jid}


@app.get("/api/rebuild/{jid}")
def status(jid: str):
    job = _job(jid)
    return {"status": job["status"], "log": job["log"][-60:]}


@app.get("/api/rebuild/{jid}/image")
def image(jid: str):
    job = _job(jid)
    if job["status"] != "done":
        raise HTTPException(409, "not finished yet")
    return FileResponse(job["image"], media_type="image/jpeg")
