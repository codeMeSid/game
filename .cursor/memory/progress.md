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

---

## Plan addendum — start/finish line, lap success, loop snap, arrows, race countdown

### Goal

1. **One start/finish line** — Single segment (or aliased `startLine` === `finishLine`) used for laps; update UI from separate START/FINISH to one **Start/finish line** control.
2. **Ghost / training success** — Success requires **traversing the track** and **crossing the start/finish line** in the forward direction (with progress / debouncing to avoid false positives).
3. **Arrow keys** — Primary driving input for player (document; optional: keep WASD as secondary).
4. **Closed loop snap** — When first and last centerline points are within a threshold, snap together for a continuous closed circuit.
5. **Race mode** — **Start race** button → **countdown** overlay (e.g. 3-2-1-GO) → both cars run; ghost tries to win (baseline or trained policy).

### Implementation sequence

1. **Track schema (`index.html` + backend payload):** Introduce `startFinishLine` (two points); migrate loaded data from `startLine`/`finishLine` to a single line; remove duplicate markers in UI.
2. **`snapLoopIfNeeded()`** after stroke commit (and optional button): if `dist(center[0], center[last]) < threshold`, align last to first (or average) and rebuild walls.
3. **Lap math:** Arc-length along centerline; max progress per episode; segment intersection with `startFinishLine` + forward tangent test; debounce lap increments.
4. **Ghost baseline + training:** Success when lap completes; backend `main.py` reward/termination aligned with same geometry and progress rules.
5. **Controls:** Filter `stepPlayer` to **arrow keys** (optionally retain WASD).
6. **Race state machine:** `idle` → `countdown` → `racing`; **Start race** button; overlay shows numbers; zero motion during countdown; on GO enable player + ghost competition; optional “winner” when first to complete N laps or cross after full progress.

### Verification

- One visible line for start/finish; saved JSON has a single logical line.
- Ghost/training registers success only after a valid lap (closed loop) or documented fallback for open tracks.
- Arrows drive the car; snap visibly closes gaps when endpoints meet.
- Race only starts after countdown; ghost races for win once **GO** shows.

---

## Plan addendum — backend-guided training connection + visibility rules

### Goal

- Make training strictly **backend-guided**.
- If backend is unavailable/disconnects, training must auto-stop.
- During training and end-session, player car is hidden (training view is backend ghost only).

### Frontend state model (`index.html`)

- Add explicit state:
  - `backendConnection: disconnected | connecting | connected | error`
  - `trainingState: idle | starting | running | stopping | stopped | error`
- Transition helpers:
  - `startTrainingFlow()`
  - `stopTrainingFlow(reason)`
  - `onBackendDisconnect(reason)`

### Connection / training contract

1. `POST /session`
2. `POST /session/{id}/track`
3. Open `WS /ws/session/{id}` and wait for first ack/metrics
4. `POST /session/{id}/train/start`
5. On ws `close/error` OR heartbeat timeout OR HTTP failure:
   - `trainingState -> stopped`
   - clear training sim buffer
   - update HUD to stopped/error

### Render gating rules

- `mode === train && trainingState === running`:
  - render backend ghost + rays only
  - **hide player car**
- `endSession()`:
  - call `disconnectBackend()`
  - clear training/race runtime state
  - hide cars until next valid spawn/start

### Reliability hardening

- Heartbeat watchdog: if no `metrics/sim_state` for ~1.5-2.0s during running training, auto-stop with message “backend disconnected”.
- Guard start button: disabled unless backend connected and track valid.
- `disconnectBackend()` always idempotent and forces `trainingState=stopped`.

### Verification

- Start training with backend down -> no fake/local training; status shows stopped/error.
- Kill backend mid-training -> training stops automatically; ghost/player training visuals halt.
- In training, player car is not visible.
- On end session, no car remains visible and no stale training stream continues.

---

## Plan addendum — highest training quality (crash-early fix)

### Goal

Improve backend trainer so ghost learns stable lap driving instead of brief movement + early crashes.

### Phase 1 (immediate quality gains)

1. **Upgrade trainer core** in `apps/api/main.py`
   - Proper PPO clip objective + GAE-lambda advantages.
   - Stable value/entropy losses and gradient clipping.
2. **Reward redesign**
   - Strong forward progress along centerline.
   - Large terminal reward for valid lap completion at start/finish line.
   - Collision/off-track penalties tied to impact/severity.
3. **Readiness quality gate**
   - `ready` only if rolling success rate + collision-rate + return-stability thresholds are met.
4. **Curriculum**
   - Begin with wider road + lower speed cap, then scale difficulty.

### Phase 2 (true recurrent quality path)

1. **Sequence-aware recurrent PPO**
   - Preserve hidden state boundaries and train on temporal chunks.
2. **Observation normalization**
   - Running mean/std for obs features; consistent at inference time.
3. **Checkpoints + evaluation**
   - Save periodic checkpoints, track “best model” by deterministic eval episodes.
4. **Quality presets**
   - `fast | balanced | quality` profiles for rollout steps / epochs / batch sizes.

### Frontend metrics improvements (`index.html`)

- Show quality-focused metrics during training:
  - `successRate`, `collisionRate`, `avgEpisodeLen`, `evalScore`, `bestGeneration`
- Keep `ready` badge tied to backend quality gate, not just return.

### Verification

- On a representative loop track, training transitions from early crashes to sustained lap completion.
- Success rate and episode length trend upward; collision rate trends downward.
- `ready` appears only after sustained competence window.

---

## Plan addendum — console error triage and cleanup

### Goal

Separate real app errors from extension noise and keep console actionable.

### Triage sequence

1. Re-run in a clean browser profile (or Incognito with extensions disabled).
2. If `DynamicTree.js` and “Receiving end does not exist” disappear, classify as external extension noise.
3. Keep only app-owned issues in active bug list.

### App-owned fixes

1. Add favicon reference/file to remove `:5173/favicon.ico 404`.
2. Add explicit `[wonder-car]`-prefixed logs for:
   - backend connect/start/stop failures
   - websocket close/error and heartbeat timeout
3. Add null-guards for DOM element access and idempotent state transitions where needed.

### Verification

- Clean-profile run shows no extension stack traces.
- App console contains only relevant `[wonder-car]` status/errors.
- Backend training disconnects still produce clear, single-stop transition logs.

### Status (implemented in `/develop`)

- Added inline favicon link in `index.html` to remove `:5173/favicon.ico 404`.
- Added `[wonder-car]`-prefixed frontend logging (`info/warn/error`) for backend connect/train/ws flows.
- Hardened frontend backend calls with explicit `res.ok` checks and clear failure logs.
- Added websocket `onerror` handling and ensured `onclose` resets training status/buffer.
- Added required DOM node guard at startup to fail fast with clear app-specific error.
- Fixed backend warning in `apps/api/main.py` by switching to `adv.std(unbiased=False)`.

### Status (implemented in `/develop` — UX overhaul batch)

- Removed **Connect backend**; **Start/Stop training** single toggle; compact **Backend:** pill (text, not a large circular light).
- Renamed **Drive (random)** → **Practise**; random AI only in Practise; player car **inactive/hidden** in Draw & Train (active only Practise/Race).
- **Larger car** fixtures; **smoother steering** (slew + lower torque); **sub-stepped** `world.step`; stronger **lateral grip vs speed**.
- **Race-themed** sharp UI (monospace, borders, accent).
- HUD: **You / Ghost km/h**; training block uses plain-language labels (update step, refinement pass, practice steps, success rate, score).
- Backend: slightly **softer** training sim (steer rate, wall damage), longer rollouts, lower LR.

---

## Plan addendum — UX overhaul, physics, practise/race, training HUD

### Issues to address

1. Remove **Connect backend** button; replace with **auto-connect on train start** and a **small** backend status indicator (not a large circular signal).
2. **Steering** too snappy → rate-limit steering input, lower torque gain, optional angular velocity clamp.
3. **Car jumping** → sub-step physics, clamp forces/torque, verify no duplicate force application; ghost inactive when unused.
4. **UI** → race theme: sharp panels, condensed typography, accent borders, carbon/dark base.
5. **Post-turn acceleration slide** → stronger speed-dependent lateral friction; smooth throttle; cap combined slip.
6. **Training quality** → continue PPO/GAE + reward/curriculum in `apps/api/main.py`.
7. **Larger car** → increase Planck box half-extents + matching `renderCar` / nose graphic scale.
8. Rename **Drive (random)** → **Practise** (keep or alias mode id).
9. **Single Start/Stop training** button (toggle); remove separate connect flow from UI.
10. **Random/practise car** visible only in **Practise** and **Race** (hide in Draw/Train per product rules).
11. HUD: **Player speed** + **Ghost speed** (same units, e.g. km/h or “u/s”).
12. Training HUD: human labels — e.g. “Learning step (generation)”, “Success rate”, “Avg episode length”, “Crashes”, tooltips for jargon.

### Implementation order (`index.html`)

1. CSS: new race theme variables, panels, typography; compact backend status pill (replace any circular motif).
2. Remove connect button; add `trainToggle` wired to connect-if-needed + start/stop; update DOM guards.
3. Rename mode button Practise; gate “random drive” visibility and AI to Practise + Race only.
4. Physics tuning: larger fixture, steering slew, torque/force limits, optional substeps, lateral friction vs speed.
5. HUD: speed readouts + training plain-language stats block.
6. Backend: training quality tranche (see prior addendum).

### Verification

- Training starts without Connect button; status visible but minimal.
- Practise/Race: stable driving, no jitter; post-turn acceleration does not instantly throw car sideways.
- Train: ghost only; no random car; player hidden as before.
- Speed numbers track movement; training labels readable without ML jargon.

---

## Plan addendum — Practise flicker/spin + console error explanation

### Console errors (understanding)

| Message | Typical cause |
|--------|------------------|
| `Could not establish connection. Receiving end does not exist.` | **Browser extension** messaging (e.g. ad blocker, password manager, Cursor/dev tools helpers). Not from wonder-car unless we call extension APIs. |
| `DynamicTree.js ... setting 'height'` | **Third-party injected script** (often extension). Not a file in this project. |

**Verify:** Incognito / extensions off → if errors disappear, ignore for app debugging.

### Practise issues — fix plan (`index.html`)

1. **Circles / “random” driving:** Practise currently applies **sinusoidal auto steer** → natural **orbits**. Change default Practise to **arrow-keys only**; add optional **Auto demo** toggle with softer, bounded steering.
2. **Flicker / “multi cars”:** Guard **`spawnCarsOnTrack()`** so it does not run on redundant `setMode` (same mode); only on mode **change** or after **track commit**. Wrap **`uploadTrackToBackend()`** in try/catch on canvas handlers.
3. **Crashes:** With manual default + softer auto demo + existing traction limits, wall hits should drop.

### Verification

- Practise: one car, stable spawn, no strobing when switching modes slowly.
- Auto demo optional and clearly labeled.
- Clean browser profile: no `DynamicTree.js` / receiving-end spam from extensions.

### Status (implemented in `/develop` — Practise + upload hardening)

- **Practise** defaults to **manual** (arrows); **Auto demo (AI steering)** checkbox enables softer sin demo; disabled when not in Practise.
- **`setMode`**: **`spawnCarsOnTrack()`** only when `prev !== next` (avoids flicker on repeat mode clicks).
- **`uploadTrackToBackend()`**: try/catch, **`logWarn`**, returns boolean; no unhandled rejections from failed uploads.
- HUD hint updated for Practise / Train.

---

## Plan addendum — Backend pill “always offline”

### PM — scope & acceptance criteria

**Problem:** The status pill shows **Backend: offline** even when the FastAPI server is running, because **offline** today means “no WebSocket session yet” (`!connected && !backendSessionId`), not “API unreachable.”

**Goal:** Users can tell at a glance whether the **Python API is reachable** vs **disconnected / training**.

**Acceptance criteria**

1. With **uvicorn** running on the configured `baseUrl`, the pill shows something other than **offline** (e.g. **online** / **idle**) **before** the user clicks **Start training**, once a health check succeeds.
2. With **no server** on `baseUrl`, the pill shows **offline** (or **unreachable**) after the health check fails.
3. **Start training** still transitions to **connected** / **training** as today; WebSocket URL stays aligned with **`state.backend.baseUrl`** (no hardcoded host/port mismatch).
4. **CORS:** `GET /health` works from the same origins already used for `POST /session` (extend `allow_origins` if a new dev origin appears, e.g. another port).

### Architect — design & file changes

| Area | Change |
|------|--------|
| **`apps/api/main.py`** | Add **`GET /health`** returning a small JSON payload (e.g. `{"ok": true}`). Reuse existing **CORSMiddleware** (no extra middleware unless needed). |
| **`index.html`** | Introduce **`state.backend.apiReachable`** (boolean), set by **`pingBackendHealth()`** (`fetch(`${baseUrl}/health`)` with try/catch). Call **on load** and on an **interval** (e.g. 20–30s) and optionally **on `window.focus`** to recover after laptop sleep. |
| **`index.html`** | Add **`httpBaseToWsUrl(base)`** (or inline): `http`→`ws`, `https`→`wss`, same host/port as `baseUrl`. Replace hardcoded `` `ws://localhost:8000/ws/...` `` with the derived URL. |
| **`updateBackendPill()`** | New precedence: if **API unreachable** and no session → **offline**; if **API reachable** but no **session/ws** → **online / idle** (copy TBD); keep **training**, **connected/ready**, **reconnect…** as today. |
| **Optional (nice)** | Set **`connected = true`** only in **`ws.onopen`** (not immediately after `new WebSocket(...)`) so the pill does not briefly lie “connected” before the handshake fails. |

**Implementation order**

1. Backend `GET /health`.
2. Frontend `httpBaseToWsUrl` + WebSocket construction.
3. `pingBackendHealth` + `apiReachable` + pill logic.
4. `ws.onopen` / connection flags (optional polish).
5. Manual verify: server down, server up idle, start training, stop training.

### Devil’s advocate — risks & mitigations

| Risk | Mitigation |
|------|------------|
| **CORS** missing origin (e.g. new dev port) | Add origin to `allow_origins` or document `VITE_*` / query-string base URL later. |
| **Health spam / battery** | Interval 30s+; pause or slow when tab hidden (`document.visibilityState`). |
| **False “online”** if health is proxied but WS blocked | Rare; document that training still requires WS; pill can show **online** + **reconnect…** after failed train. |
| **`[::1]` IPv6 hosts** | Add to CORS if needed. |

### Verdict

**CONCERN** — Main issue is **semantic** (offline = no WS) plus **hardcoded WS URL**; **APPROVE** plan with CORS vigilance and optional `onopen` fix.

### Verification

- Backend stopped → pill **offline** after ping.
- Backend up, page idle → pill **not** stuck on **offline**.
- Start training → **connected** / **training**; stop → returns to idle/reachable, not false offline.

### Status (implemented — training signal upgrade, `apps/api/main.py`)

- **Closest point on polyline** for tangent + arc-length `s` (replaces nearest-vertex heuristic).
- **Spawn heading** aligned with first centerline segment.
- **Reward:** arc-length progress term + small backward penalty; **finish line** segment crossing gives large bonus and episode reset; success metric includes **lap finished**.
- **Observation:** normalized **progress** along track (`obs_dim` 15).
