# drexathon

Workspace for the drexathon project.

## Live demo

**https://unshred-production.up.railway.app**: upload a photo of torn or shredded paper
pieces and watch Drex Max rebuild the page, step by step (find, paper check, pair ratings,
neighbour picks, layout, piece agents, final review), with timings and a history of runs.

Run locally: `uv run --env-file .env uvicorn unshred.webapp:app --port 8765`
(needs `NACE_API_KEY` in `.env`). Results and benchmarks: `docs/RESULTS.md`.
