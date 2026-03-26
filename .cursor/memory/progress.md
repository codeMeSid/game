# Progress — wonder-car

## Current state

- `/init` completed: project identity (`.cursor/rules/soul.mdc`), memory files, and subagent skill scaffolds created.
- `/plan` completed for single-HTML MVP (track drawing + physics + RL ghost + racing).
- `/plan` completed for adding **local Python backend** to implement **true RNN** recurrent RL.
- `/develop` in progress: scaffolded `index.html` + Python backend, installed deps, and smoke-tested `POST /session`.

## Plan — single HTML MVP (deliverable: `index.html`)

### Goal

Create a **single self-contained** front end with a **16:9 canvas** where users can draw a track, persist it for the duration of a session, drive a tiny car with physics + health/damage, train a “ghost” car (RL, recurrent if feasible), and race against it.

### Constraints / decisions

- **Single file**: `index.html` only.
- **Physics**: use a 2D simulator via pinned CDN (Planck.js preferred).
- **Session persistence**:
  - `sessionStorage` holds `sessionId` (tab session).
  - Track data stored in `localStorage` keyed by `sessionId`.
  - If `sessionStorage` missing, treat as session ended and clear stored track for that session.
- **RL risk mitigation**:
  - Provide baseline “ghost” (centerline follower) so race mode works even if RL is slow/experimental.
  - Training runs in a **Web Worker** to keep UI responsive.

### Step-by-step implementation sequence

1. **Scaffold `index.html` shell**
   - Canvas (16:9), HUD, mode buttons: Draw / Drive(Random) / Train(Ghost) / Race / End Session.
   - Render loop (requestAnimationFrame) with fixed physics timestep accumulator.

2. **Session + persistence**
   - On first interaction, create `sessionId`, store in `sessionStorage`.
   - Save/load track state (polyline points, road width, start/finish line endpoints).
   - “End session” clears session + stored track and resets UI.

3. **Track drawing UX**
   - Draw centerline polyline with mouse/touch; decimate points (min distance).
   - Smooth (optional) and enforce minimum segment length.
   - Derive left/right boundaries by offsetting normals using roadHalfWidth.
   - Render road fill + boundaries for visual clarity.

4. **Physics world + walls**
   - Initialize Planck world.
   - Create static colliders from left/right boundary polylines (edge segments).
   - Add debug toggle to render collider lines and contacts.

5. **Car physics + controls**
   - Dynamic body for player car; apply throttle/steer forces.
   - Implement “random drive” controller as baseline.
   - Clamp max speed; stabilize with lateral friction approximation.

6. **Health + collision damage**
   - Track `health` in [0..100].
   - On wall contact, compute damage based on contact impulse or impact speed.
   - When `health <= 0`: freeze controls, spawn blast particles, show centered “GAME OVER”.

7. **Ray casting sensors**
   - Implement N rays (e.g. 11) fan from car heading; physics raycast to walls.
   - Render rays and hit points; expose normalized distances as observation vector.

8. **Ghost car baseline**
   - Implement a simple centerline-follow controller using lookahead point + PID-like steering.
   - Use it for initial “Race” mode so the app is fun before RL works.

9. **Training architecture (Worker)**
   - Build worker that can:
     - Simulate episodes in a fast, headless stepping loop (either same physics lib in worker or simplified kinematics).
     - Output periodically: policy snapshot + metrics (episode return, completion rate).
   - UI shows **status**: `in-training` vs `ready` based on metric thresholds.
   - Visualize training ghost in main thread using latest policy snapshot; draw rays while training.

10. **Race mode**
   - Two cars: player + ghost.
   - Decide collisions: disable car-car collision for stability; both collide with walls.
   - Modes: solo (ghost hidden), vs ghost (both visible).

11. **Polish**
   - Clear visual states: training banner, ready badge, health bar, minimap-ish track rendering.
   - Parameter panel (road width, car size, sensor count, damage multiplier) stored per session.

## Blockers / known risks

- True **recurrent RL** inside a single HTML (performance + stability). Baseline ghost mitigates this.
- Boundary offset self-intersections on tight turns; requires smoothing and constraints to avoid broken walls.

## Blockers

- None.

---

## Plan — local Python backend for true RNN recurrent RL

### Goal

Add a local-only Python service that can train a **true recurrent** policy (GRU/LSTM) and stream metrics/status to the frontend, without making the frontend depend on per-frame HTTP calls for control.

### Key constraints / decisions

- **Local-only**: `localhost` only; no auth; no DB.
- **Do not do per-frame HTTP inference** (latency). Prefer:
  - backend = training + metrics, and
  - inference = local (later), or low-rate WebSocket inference if needed temporarily.
- Start with a **defined environment** that matches the frontend track format, even if dynamics are simplified at first.

### Proposed monorepo shape

- `apps/web/` — React + Vite UI (later; not required for the single-HTML prototype).
- `apps/api/` — Python FastAPI service.
- `packages/shared/` — shared JSON schema/types for track/session payloads.

### Step-by-step implementation sequence (backend + integration)

1. **Define shared payload schema**
   - Track centerline points, road width, start/finish line endpoints.
   - Session identifiers and training params (sensor count, timestep, max steps, reward weights).

2. **Backend scaffold (FastAPI)**
   - HTTP: create session, upload track, start/stop/reset training.
   - WebSocket: stream `{status, metrics}` updates.
   - CORS limited to local dev origins.

3. **Environment implementation (Python)**
   - Minimal 2D car model for training (kinematic bicycle or simple point-mass with steering).
   - Ray casting against track boundaries derived from the same offset logic as frontend (or directly from provided boundaries).
   - Episode termination: finish line crossed, max steps, or “health” exhausted due to boundary impacts.

4. **True recurrent policy (PyTorch)**
   - GRU/LSTM with hidden state carried across steps.
   - Actor-critic heads (policy + value).
   - PPO training loop with GAE; deterministic seeding for reproducibility.

5. **Status + readiness**
   - Maintain moving windows for return and success rate.
   - Emit `in-training` vs `ready` based on thresholds and stability over N episodes.

6. **Frontend integration (single HTML phase)**
   - Add “Connect backend” and “Start training” controls.
   - Upload track payload on session start or when track changes.
   - Subscribe to WS metrics; show training status/ready badge.
   - Keep driving/racing functional with baseline ghost regardless of backend availability.

7. **Inference strategy (choose one)**
   - **Preferred** (later): export trained policy to a format runnable in browser (e.g. ONNX) and run local inference.
   - **Interim**: WebSocket inference at 20–30 Hz with client-side smoothing; reset hidden state at episode start.

### Major risks to watch

- Sim-to-real mismatch (Python kinematic env vs JS physics).
- RNN hidden-state reset/serialization bugs.
- PyTorch install friction; keep setup docs lean and local-only.

---

## Plan addendum — live training simulation in frontend

### Goal

While backend training runs, the frontend should show a **live training simulation** (ghost car moving, raycasts visible), not just metrics.

### Approach (MVP)

- Use the backend as the **authoritative simulator** during training.
- Stream a **live preview** over WebSocket:
  - `metrics` at ~2–5 Hz (loss/return/success/status).
  - `sim_state` at ~10–20 Hz (ghost pose + optional ray endpoints).
  - Optional `sim_chunk` buffering (0.5–1.0s of future states) to smooth jitter.
- Frontend renders the ghost using **interpolation** between buffered states in its 60fps render loop.

### Implementation steps

1. Extend backend WS to emit `sim_state` (and optionally `sim_chunk`) for the currently running training episode.
2. Add frontend WS handler + ring buffer for ghost states.
3. Render ghost pose from buffered states; draw rays either from streamed endpoints or recompute against local track geometry.
4. Add UX states: `buffering`, `in-training`, `ready`, `stopped`.

### Guardrails

- Cap stream rate and payload size (quantize, reduce rays) to prevent bandwidth spikes.
- Keep gameplay/race independent: if backend disconnects, fall back to baseline ghost.

---

## Plan addendum — training epoch + generation counters

### Goal

While training, show **Epoch** and **Generation** counters in the frontend HUD.

### Semantics (MVP, PPO-aligned)

- **Generation** = one policy update step (after collecting a rollout).
- **Epoch** = one optimization pass over the rollout batch during that generation (1..`ppo_epochs`).

### Protocol changes

Add these fields to each WebSocket `metrics` message:

- `generation` (int), `generation_total` (optional int)
- `epoch` (int), `epoch_total` (optional int)
- Also include `timesteps` (int) for an always-increasing progress metric

### Frontend UI

- Training HUD displays:
  - `Status: in-training | ready`
  - `Generation: g[/G]`
  - `Epoch: e[/E]`
  - `Timesteps: t`

---

## Plan addendum — frontend bugfixes (game over, spawn, session, rays)

### Issues to fix

1. **GAME OVER overlay stuck** — add explicit **Restart** / dismiss that clears `gameover`, hides overlay, and re-inits physics (`resetWorld()`).
2. **Cars appear random / move wrong** — consolidate canvas pointer handlers; fix ghost vs physics step order; keep pixel ↔ Planck meter scaling consistent.
3. **Starting line not defined** — default `startLine` (and optionally `finishLine`) when track is first committed; add UI hints or buttons for “Set START / FINISH”.
4. **End session does nothing** — `endSession()` must: close WebSocket, clear backend ids/buffers, clear storage, `resetWorld()`, refresh HUD.
5. **Race spawn off track** — `spawnCarsOnTrack()` using start-line midpoint + tangent, else centerline[0]→[1].
6. **Rays only in training** — draw sensor rays only for **training ghost** in **Train** mode; never for player in Drive/Race.

### Implementation order (`index.html`)

1. Add overlay actions: **Restart** + optional **Dismiss**; ensure `resetWorld` clears `overlayEl.style.display = "none"`.
2. Merge pointer logic so draw vs start/finish placement don’t fight.
3. After decimated centerline has ≥2 points, set default start/finish segments if missing; `saveTrack()`.
4. Implement `spawnCarsOnTrack()`; call from `resetWorld` and when switching to Drive/Race/Train if track exists.
5. Harden `endSession()`: `ws.close()`, reset `state.backend`, clear sim buffer, then storage + `ensureSession` + `resetWorld`.
6. Gate rendering: `castRays(playerBody)` removed; ghost rays only when `mode === Modes.TRAIN` (and training active / backend as designed).

### Verification

- Die → Restart → overlay gone, car moves again.
- End session → blank track, no ghost WS, no stale HUD.
- Draw track → cars spawn on road; start line visible (default or user-set).
- Drive/Race → no blue ray fan; Train → ghost rays only.

### Status (implemented in `/develop`)

- **Restart** button on GAME OVER + `restartGame()` clears overlay and respawns on track.
- **`disconnectBackend()`** + hardened **`endSession()`** (WS close, backend state reset, then storage + `resetWorld()`).
- **`ensureDefaultStartFinish()`** after stroke commit and on load; **Set START / FINISH line** UI + merged pointer flow (no competing `pointerdown` handlers).
- **`spawnCarsOnTrack()`** / **`getSpawnPose()`** from start-line midpoint + tangent.
- **Player raycasts removed**; training ghost rays only in **Train** mode (backend rays or fallback `castRays`).
- **Ghost** inactive during training preview (kinematic `setTransform`); **Race** baseline ghost still uses physics when backend disconnected.
- **Z-index**: HUD above game-over overlay so **End session** stays clickable.
