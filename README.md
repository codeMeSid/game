# wonder-car

Local-only prototype: draw a track, drive a tiny car with physics + health, and watch a “ghost” training simulation stream live from a Python backend (with **epoch**/**generation** counters).

## Run (frontend)

- Open `index.html` in a browser (best: a local static server).

Example:

```bash
python3 -m http.server 5173
```

Then visit `http://localhost:5173/`.

## Run (backend)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn apps.api.main:app --reload --port 8000
```

Backend listens on `http://localhost:8000`.

## Notes

- The frontend stores the current drawing in `localStorage` keyed by a `sessionStorage` session id (so it persists until you “End session”).
- Training preview is streamed over WebSocket; the frontend interpolates snapshots for smooth rendering.

