# Code conventions — wonder-car

Detected / agreed patterns (workspace was bootstrapped empty; update as the repo grows).

## Repository

- **Monorepo:** expect `apps/` (e.g. Vite app) and `packages/` (shared types, sim, RL) once scaffolded.

## Runtime prototype

- Single-file prototype lives at repo root as `index.html` for fastest iteration.
- Python backend lives in `apps/api/` (FastAPI + WebSocket).

## Frontend (`index.html`)

- **Sensor rays** are drawn only for the **training ghost** in **Train** mode, not for the player car.
- **START/FINISH**: default lines are created when a stroke has ≥2 points; user can override via **Set START line** / **Set FINISH line** (click-drag on canvas).
- **Spawn**: player and ghost use **`spawnCarsOnTrack()`** from the start-line midpoint (or first center segment) with heading along the track.
- **End session** calls **`disconnectBackend()`** then clears storage and **`resetWorld()`**.
- Use `[wonder-car]` log prefix for app-owned console diagnostics so extension/browser noise is easy to distinguish.
- **Backend status** is a compact text pill (`Backend: offline | connected | training | ready`), not a large circular indicator.
- **Practise** mode: default **arrow keys**; optional **Auto demo (AI steering)** (`#chkAutoDemo`, `state.practise.autoDemo`) uses soft bounded sin steering—disabled outside Practise. Player body is physics-active only in **Practise** and **Race**; inactive in **Draw** and **Train**.
- **`setMode`**: call **`spawnCarsOnTrack()`** only when the mode **value** changes (`prev !== next`), not on redundant clicks; track edits still spawn via **`commitStroke`** / line placement / **`resetWorld`** / **`restartGame`**.
- **`uploadTrackToBackend()`**: returns boolean; failures **`logWarn`** (no throw); callers tolerate offline backend.

## TypeScript / React

- Strict TypeScript; prefer explicit types at module boundaries.
- Functional React components; hooks for state and effects.
- Co-locate feature code where it keeps imports short; extract shared pieces to `packages/` when reused.

## Formatting & lint

- **All** tooling as chosen in root config (ESLint + Prettier typical for TS/React). Run format/lint after meaningful edits.

## Testing

- **None** initially; when added, prefer one runner (e.g. Vitest) aligned with Vite.
