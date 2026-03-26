# Council session — latest

Date: 2026-03-26
Topic: Single-HTML canvas track editor + physics driving + RL ghost training + race mode

## PM Agent

### Scope

- Single `index.html` front end only (no build step required) with a **16:9 canvas**.
- Track drawing/tracing on canvas that creates a **session**; the track **persists until session end**.
- Modes:
  - **Draw**: create/edit track.
  - **Drive (random)**: tiny car drives with random control options (baseline).
  - **Train (ghost)**: train an intelligent ghost car via **recurrent reinforcement learning**; visualize **ray casting sensors** while learning. Show training iterations.
  - **Race**: race human vs ghost or solo.
- Physics + collision:
  - Use a 2D physics simulator.
  - When car hits road boundary, **decrease health based on impact force/speed**.
  - On 0 health: **blast animation** + **GAME OVER** centered overlay.
- Training status:
  - Show **in-training** while learning.
  - Show **ready** when confidence is good.
  

### Acceptance criteria (MVP)

- Canvas is 16:9 and responsive; track drawing is usable.
- Track persistence works across reloads during an active session; “End session” clears it.
- Car can drive and collide; health decreases proportionally; gameover + blast visible at 0.
- Ghost training runs without freezing UI; rays draw each frame; status flips to ready after meeting a measurable threshold.
- Race mode renders two cars with distinct styles; ghost follows learned policy.

### Product risks

- “Recurrent RL” in pure JS is complex and may be unstable; scope creep risk is high.
- Training time may be long without GPU acceleration; needs pragmatic “good enough” definition.
- Track drawing UX can balloon (undo/redo, smoothing, walls thickness).

## Architect Agent

### Technical design (single HTML)

- **Single file**: `index.html` containing:
  - Canvas + simple UI (mode buttons, session controls, status HUD).
  - JS modules embedded via `<script type="module">`.
- **Physics**: use a lightweight 2D physics engine from CDN:
  - Prefer **Planck.js** (Box2D port) for stable rigid-body collisions and contact impulses.
  - Alternative: Matter.js (easier, but impulse/contact forces are less direct).
- **Track representation**
  - User draws a centerline polyline; derive **left/right boundaries** by offsetting normals by roadHalfWidth.
  - Convert boundaries into static physics colliders:
    - Simplest: many small segments (edge shapes) for each boundary polyline.
  - Start/finish: draggable line segments (two points each) snapped to track vicinity.
- **Session persistence**
  - `sessionId` created on first draw (UUID).
  - Store `{sessionId, createdAt, trackPoints, roadWidth, startLine, finishLine}` in `localStorage`.
  - “End session” button clears storage + resets state.
  - Note: “until session end” interpreted as localStorage persistence until user explicitly ends.
- **Car model**
  - Dynamic body (rectangle) with:
    - `throttle` (forward force), `steer` (angular impulse/torque), optional drift clamp.
  - Health system:
    - On collision contact, read normal impulse (Planck contact impulse) or approximate via relative velocity.
    - Damage = `k * max(0, impulse - threshold)` or `k * speed^2` at impact.
  - Blast animation:
    - Particle burst (circles) spawned at car position; fades over ~1s.
    - Overlay text “GAME OVER” centered; pause sim.
- **Ray casting sensors**
  - From car pose, cast N rays in a fan (e.g. 9–21) using physics raycast against track walls.
  - Observations: normalized distances + car speed + heading error to local track tangent.
  - Visualize rays each frame for ghost (and optionally player).
- **RL training (pragmatic plan)**
  - Run training in a **Web Worker** to keep UI responsive.
  - Start with a recurrent policy that’s feasible in JS:
    - **Option A (pragmatic “recurrent”):** use a small GRU/LSTM via **TensorFlow.js** (CDN) in the worker, train with a simple policy gradient (REINFORCE) or PPO-lite.
    - **Option B (lighter):** treat recurrence as finite history window (stack last K observations) and use a non-recurrent network (still “recurrence-like”); label as “recurrent” only if true RNN is implemented.
  - Reward shaping:
    - Positive for forward progress along centerline.
    - Negative for wall hits, high steering jitter, and time penalty.
    - Big positive for crossing finish line; terminate episode on health 0 or off-track.
  - “Confidence / ready” metric:
    - Moving average of episode return and completion rate; ready when above thresholds for N episodes.
- **Modes**
  - Drive (random): random actions at fixed interval; good baseline + debugging.
  - Race: simulate two cars in same world or two parallel sims; simplest is same world with collision enabled between cars.

### Affected files

- `index.html` only (single-file deliverable).

### Implementation sequence (high-level)

- Build canvas + render loop; implement input handling and mode switching.
- Implement track drawing + boundary generation + persistence.
- Add physics world and player car; collisions → health → explosion/gameover.
- Add ray casting visualization.
- Add ghost car with heuristic baseline (follow centerline) to validate sensors.
- Add worker-based training loop + status + “ready” gating.
- Add race mode + UI polish.

## Devil’s Advocate

### Critiques / edge cases

- Offsetting a drawn polyline to generate boundaries can self-intersect on sharp turns; collisions become unstable.
  - Mitigation: smooth/decimate points; enforce minimum curvature; prevent too-tight turns; or draw boundaries directly (two-lane lines).
- Planck edge-segment chains can snag car corners (“edge ghost collisions”) if segments aren’t welded/continuous.
  - Mitigation: chain shapes carefully; keep segment length small; consider thicker walls with polygons.
- “Recurrent RL” with tfjs in a worker is heavy and may still be slow on CPU; training might feel broken.
  - Mitigation: ship a **baseline ghost** first (centerline controller), then add RL as “experimental.”
- Reward shaping can lead to degenerate behaviors (spinning to farm reward, hugging walls).
  - Mitigation: progress computed by projection onto centerline arc-length; penalize lateral deviation and wall proximity.
- LocalStorage “session” can persist longer than intended; user expects true session (tab lifetime).
  - Mitigation: store `sessionId` in `sessionStorage` and track data in `localStorage` keyed by id; if `sessionStorage` missing on reload, treat as ended and clear.
- Security / supply-chain: pulling physics/ML libs from CDN is riskier.
  - Mitigation: pin versions + use reputable CDNs; later vendor locally.

## Verdict

**CONCERN** — feasible as a plan, but RL scope is the main risk. Proceed with mitigations:
- Define an MVP where training can be “good enough” and provide a non-RL baseline ghost.
- Treat “session end” as tab session by using `sessionStorage` as the authoritative session token.

---

Date: 2026-03-26
Topic: Add local Python backend for true RNN recurrent RL

## PM Agent

### Scope

- Add a **local-only Python backend** that:
  - Trains a **true recurrent** policy (GRU/LSTM) for the ghost car.
  - Streams training status/metrics to the frontend.
  - Serves the latest policy for inference during “Train” (preview) and “Race”.
- Keep frontend focused on UX/physics/rendering; backend owns training compute.

### Acceptance criteria (MVP)

- One command starts backend; frontend connects and shows:
  - **in-training** status with live metrics.
  - **ready** when threshold reached.
- Ghost car in frontend can query/use the latest policy for actions.
- Training can be stopped/reset per session.

### Product risks

- Cross-platform setup complexity (Python env, deps) can slow iteration.
- If physics differs between backend and frontend, learned policy may transfer poorly.

## Architect Agent

### Design choice: where simulation runs

- **Preferred**: backend runs training on the **same environment logic** as frontend (shared track format + consistent dynamics).
  - In practice, easiest is:
    - Backend uses a simplified but well-defined kinematic model for training (fast, stable).
    - Frontend uses physics engine for visuals + collisions; we accept a sim-to-real gap initially.
- Long-term: unify by moving physics to a common library (WASM/Box2D) or by having backend be the authoritative sim and frontend be a renderer.

### Backend stack

- **FastAPI** for HTTP + WebSocket.
- **PyTorch** for true recurrent networks (GRU/LSTM) and RL (PPO/A2C).
- Local-only CORS allowed to `http://localhost:*`.

### API shape (minimal)

- **HTTP**
  - `POST /session` → create backend training session, returns `sessionId`.
  - `POST /session/{id}/track` → upload track geometry + start/finish + params.
  - `POST /session/{id}/train/start` / `stop` / `reset`.
  - `GET /session/{id}/policy` → returns serialized weights snapshot or a compact policy id.
- **WebSocket**
  - `/ws/session/{id}` emits metrics: episode return, success rate, loss, status (`in-training`/`ready`), and optionally a small set of trajectories for debugging.

### Model / RL details (MVP)

- Observation: ray distances + speed + heading error + maybe last action.
- Policy: GRU (hidden state carried step-to-step) → action mean (throttle, steer) + value head.
- Algorithm: PPO (stable baseline).
- “Ready” definition: moving average success rate and return above thresholds for N episodes.

### Monorepo layout (once code exists)

- `apps/web/` (React + Vite UI)
- `apps/api/` (Python FastAPI)
- `packages/shared/` (track/session JSON schema, constants)

### Implementation sequence

- Define shared JSON schema for track/session.
- Implement backend session + websocket metrics.
- Implement frontend connector (create session, upload track, start/stop training, poll/subscribe to status).
- Implement inference path: frontend sends observation (+ hidden state) and receives action, or downloads model snapshot and runs inference locally (later).

## Devil’s Advocate

### Critiques / failure modes

- **Latency**: if frontend requests actions over HTTP each frame, controls will be laggy.
  - Mitigation: only use backend for training; for gameplay inference, prefer running policy locally (WASM/JS) or batch actions.
  - Alternative: keep a WebSocket open and send/recv at fixed 20–30 Hz.
- **Hidden state handling**: RNN needs consistent reset at episode boundaries; bugs will silently ruin learning.
  - Mitigation: explicit `reset_hidden` at env reset; include hidden state in protocol if inference is remote.
- **Sim mismatch** between backend training and frontend physics will reduce performance.
  - Mitigation: start with kinematic model in both training and inference; keep frontend physics as “rendered approximation” until unified.
- **Dependency weight**: PyTorch is large.
  - Mitigation: keep local-only; document setup; optionally use `uv` or `poetry` later.

## Verdict

**CONCERN** — backend enables true RNN RL, but we must avoid per-frame network inference. Proceed with:
- Backend for training + metrics.
- Frontend uses baseline ghost for racing until local inference is available (or websocket inference at low Hz).

---

Date: 2026-03-26
Topic: Live training simulation visible in frontend

## PM Agent

### Scope / user impact

- While training runs, user should see the **ghost car driving live** on the drawn track, with:
  - Raycasts visible from the ghost car
  - Status updates (in-training → ready)
  - Live metrics (episode, return, success rate)
- User can still interact (pan/zoom optional, stop training, switch view).

### Acceptance criteria (MVP)

- Training begins and within seconds the user sees motion (not just numbers).
- Visual playback is smooth (target ~60fps render), even if training is slower.
- “Stop training” halts updates immediately and returns control.

## Architect Agent

### Two viable approaches (choose A first)

**A) Stream trajectories/snapshots from backend (recommended MVP)**

- Backend simulates training episodes and periodically emits:
  - **Low-rate state snapshots** (e.g. 10–20 Hz): position, heading, speed, health, rays (or enough to recompute rays), plus `episodeId` and `t`.
  - Optional: short **trajectory chunks** (e.g. next 0.5–1.0s of states) to buffer jitter.
- Frontend renders the ghost using these streamed states (interpolating between snapshots).
- Benefits: frontend shows *exactly what the backend is training on*; no per-step inference calls.

**B) Stream policy checkpoints; run a live “demo sim” in frontend**

- Backend trains and occasionally sends model checkpoints (or a compact distilled policy).
- Frontend runs a separate demo simulation and drives the ghost locally.
- Benefits: smoother visuals, no network coupling; Drawback: demo diverges from training env unless physics matches.

### Protocol additions (WebSocket)

- Add message types:
  - `metrics`: rolling training stats + status.
  - `sim_state`: current live ghost state snapshot.
  - `sim_chunk`: array of future states for buffering (optional).
  - `control_ack`: start/stop/reset acknowledgements.
- Compression: keep payload small; quantize floats or reduce sensor rays if needed.

### Frontend rendering strategy

- Maintain a small ring buffer of ghost states.
- Render loop:
  - Interpolate between nearest two states by wall-clock time.
  - Draw ghost car pose + sensor rays (use provided rays or recompute via canvas/track geometry).
- When packet loss/jitter:
  - If buffer underflows, hold last state briefly; display “buffering” indicator.

## Devil’s Advocate

### Failure modes & mitigations

- **Bandwidth blow-up** if we stream full rays + many floats at high Hz.
  - Mitigation: cap stream rate (10–20 Hz), reduce ray count during training preview, or stream only endpoints.
- **Jittery motion** from variable training step time.
  - Mitigation: stream `sim_chunk` buffers and interpolate; decouple training step from streaming cadence.
- **Confusing UX** if shown sim is not the one being trained.
  - Mitigation: prefer Approach A for MVP so visuals are authoritative.

## Verdict

**APPROVE** — stream live sim snapshots/trajectory chunks over WebSocket (Approach A) and render with interpolation in the frontend.

---

Date: 2026-03-26
Topic: Show training epochs + generations in frontend

## PM Agent

### Scope / user impact

- During training, the HUD must display:
  - **Epoch** (current / total, if known)
  - **Generation** (current / total, if known)
  - Plus existing status (in-training/ready) and metrics.
- These counters should be understandable and advance predictably (no random jumps).

### Acceptance criteria (MVP)

- UI shows `Epoch: X` and `Generation: Y` updating live while training runs.
- If totals are not configured, UI shows `Epoch: X` and `Generation: Y` without totals.
- Counters reset on training reset and increment monotonically during a run.

## Architect Agent

### Define semantics (so “epoch” and “generation” aren’t vague)

- **Generation**: increments after each **policy update** (i.e. one PPO update step).
- **Epoch**: sub-iterations within a generation used to optimize on collected rollout data.
  - In PPO terms: for a given generation, we do `ppo_epochs` passes over the rollout batch.
  - Thus:
    - `generation` increments when a new rollout is collected and we start an update.
    - `epoch` increments within the PPO update loop (1..ppo_epochs).

### Protocol addition (WebSocket `metrics`)

Include in every `metrics` message:

- `generation`: integer
- `generation_total`: optional integer
- `epoch`: integer
- `epoch_total`: optional integer (often `ppo_epochs`)
- `timesteps`: total env steps consumed so far (useful secondary progress bar)
- `episode`: latest episode index (optional)

### Frontend UI

- Training HUD shows:
  - `Status: in-training | ready`
  - `Generation: g[/G]`
  - `Epoch: e[/E]`
  - `Timesteps: t`
  - existing metrics (return, success rate, loss if desired)

## Devil’s Advocate

### Pitfalls

- If we later switch algorithms (e.g. A2C), “epoch/generation” meaning changes.
  - Mitigation: keep labels generic in protocol (`update_step`, `update_epoch`) but map to UI as Generation/Epoch.
- Async streaming can show epoch > total if totals are misconfigured.
  - Mitigation: backend clamps and validates; frontend renders totals only when present and sane.

## Verdict

**APPROVE** — add epoch/generation fields to training metrics with explicit PPO-aligned semantics, and render them in the training HUD.

---

Date: 2026-03-26
Topic: Unified start/finish line, lap success, arrows-only drive, closed-loop snap, race countdown

## PM Agent

### Scope / user impact

- **Start = finish:** One **start/finish line** (single crossing segment) for the track, not two independent lines.
- **Ghost success:** The ghost must **complete the circuit** and **cross the start/finish line** in the correct direction to count as success (for training feedback and race win).
- **Driving controls:** Use **arrow keys** as the primary input for the player car (including Drive / Race); document whether WASD remains optional.
- **Closed loop:** When the drawn track’s **first and last points are near each other**, **snap** them together so the road is continuous (no gap).
- **Race flow:** In **Race** mode, user clicks **Start race** → **countdown overlay** (e.g. 3–2–1–GO) → race begins; **ghost competes** and can win.

### Acceptance criteria

- Only one line type in UI/data for the lap line (or `startLine` and `finishLine` aliased to the same geometry).
- Training / backend success metric uses **lap completion** (crossing line + minimum path progress), not only survival.
- Arrow keys move the car; HUD lists arrows as primary.
- Closing a loop within a pixel threshold snaps endpoints; walls rebuild without a visible crack.
- Race cannot begin until **Start race**; during countdown, cars frozen; after GO, player + ghost active.

## Architect Agent

### Data model

- Replace or alias **`startLine` / `finishLine`** with a single field, e.g. **`startFinishLine: [[x,y],[x,y]]`**, and migrate saved JSON (if old keys exist, copy one into the new field once).
- **Lap logic:** Store centerline as ordered polyline; compute **cumulative arc length** along centerline. Track **max arc-length progress** this episode. **Crossing detection:** segment–segment intersection of car path (or position projection) with `startFinishLine`, plus **signed side** or **direction dot product** with track tangent so crossing counts only in the forward race direction.
- **Success (ghost / training):** `progress >= trackLength * (1 - epsilon)` OR full lap: crossed start/finish after having visited most of the track (define: e.g. max s &gt; 0.85 * total length, then crossing).

### Closed-loop snap

- On stroke end (or on toggle “Close loop”): if `dist(first, last) < closeThreshold` (e.g. 24 px), set `centerline[last] = centerline[0]` (or lerp midpoint), optionally smooth one step.

### Race countdown

- States: `idle` → `countdown` (n..1) → `racing` → `finished`.
- **Start race** button sets `countdown`, disables input forces, `requestAnimationFrame` or `setInterval` for digits; on GO, set `racing`, enable ghost + player, start optional lap timer.

### Files

- **`index.html`:** Track schema, draw one line, snap closure, lap helpers, race state machine, countdown overlay, control filter (arrows).
- **`apps/api/main.py`:** Align training success / episode termination with same **start/finish** segment and **progress + crossing** (shared payload: `startFinishLine`, centerline).

### Implementation sequence

1. Schema: single `startFinishLine`; migration from legacy; `ensureDefaultStartFinish` becomes one perpendicular segment at logical start of polyline (or midpoint of closure).
2. Implement `snapLoopIfNeeded()` after stroke commit.
3. Lap progress + crossing helpers; wire **ghost baseline** and **backend reward** to “cross finish after lap.”
4. Input: arrows only (remove or gate WASD per product choice—default: **arrows primary, WASD optional**).
5. Race UI: **Start race** + countdown overlay + freeze until GO.
6. Backend: update `/track` payload and training loop success condition.

## Devil’s Advocate

- **Ambiguity:** “Start and finish the same” might mean **same segment** (approved) vs **same 2D coordinates** for two unrelated lines—clarified as one segment.
- **Self-intersecting tracks:** Lap counting can double-count crossings; mitigate with **debounce** (one crossing per lap) using a “last cross time” or **half-plane** flag.
- **Open tracks:** If user never closes loop, “full lap” may be impossible—fallback: success = reach far endpoint or crossing once with progress &gt; threshold; show UI warning “Close loop for lap racing.”
- **Countdown + physics:** Must zero velocities each frame during countdown to avoid drift.

## Verdict

**CONCERN** — proceed with mitigations: single **start/finish** segment, debounced crossing detection, optional warning for non-closed tracks, freeze physics during countdown.

---

Date: 2026-03-26
Topic: Frontend-backend training contract, visibility, and failure handling

## PM Agent

### Scope / user impact

- Frontend must clearly show how it is connected to backend for training (session + websocket lifecycle).
- During **training** and on **end-session**, the player's controllable car should not be visible.
- If backend is unavailable/disconnected during training, training must immediately stop.
- Training must be **backend-guided only** (frontend must not continue local training behavior when backend is down).

### Acceptance criteria

- Start training requires successful backend connection + track upload; otherwise shows stopped/error status.
- If websocket closes or backend calls fail while training, UI transitions to `stopped` and training visuals halt.
- In training mode, only backend ghost/sim visualization is shown; player car hidden.
- End session clears backend connection state and hides both cars until a new valid mode/start action.

## Architect Agent

### Connection flow (authoritative backend)

1. `POST /session` -> store `backendSessionId`.
2. `POST /session/{id}/track` -> upload current track.
3. Open `WS /ws/session/{id}` -> receive `metrics` and `sim_state`.
4. `POST /session/{id}/train/start` -> backend begins training; frontend state `trainingState = running`.
5. On `train/stop`, WS close, HTTP failure, or timeout heartbeat -> set `trainingState = stopped`.

### Frontend state machine

- `backendConnection`: `disconnected | connecting | connected | error`
- `trainingState`: `idle | starting | running | stopping | stopped | error`
- `renderMode`: `draw | drive | train | race`

Rules:
- `renderMode=train` + `trainingState=running` => render backend ghost sim only; hide player car.
- `trainingState!=running` => no training sim frames consumed; clear sim buffer.
- On WS close during training => auto-call local `stopTrainingUI()` (idempotent), set status/error.

### Visibility policy

- **Training mode:** hide player car entirely; show ghost + rays + metrics.
- **End session:** hide all cars until track/session reinitialized and mode allows spawn.
- **Race mode:** show player + ghost (after countdown start).

### Failure handling

- Add heartbeat watchdog: if no `metrics/sim_state` for N seconds while `trainingState=running`, auto-stop training UI and mark backend disconnected.
- Wrap all training HTTP calls with try/catch and transition to `error -> stopped`.
- `disconnectBackend()` must:
  - close WS
  - clear ids
  - clear buffers
  - force `trainingState=stopped`
  - disable training render branch

### File impact

- `index.html`:
  - explicit state machine fields + transition helpers
  - render gating (hide player in training)
  - ws heartbeat watchdog and onclose handling
  - strict backend-only training start/stop pipeline
- `apps/api/main.py` (optional hardening):
  - consistent `control_ack`/status messages for start/stop/reset to simplify frontend transitions.

## Devil’s Advocate

- If websocket drops briefly, immediate stop might feel harsh.
  - Mitigation: small grace window (e.g. 1.5s) before auto-stop.
- Hiding player in training may confuse users who want side-by-side.
  - Mitigation: optional toggle later, default remains hidden per requirement.
- Backend start may succeed but ws attach fails; frontend can look stuck.
  - Mitigation: require ws connected before allowing `train/start`, or rollback with `train/stop`.

## Verdict

**APPROVE** — implement backend-authoritative training state machine, hide player during training/end-session, and force stop on backend disconnect or heartbeat timeout.

---

Date: 2026-03-26
Topic: Training quality uplift (agent currently crashes early)

## PM Agent

### Scope / user impact

- Training quality must be significantly higher: ghost should learn stable driving and complete laps, not just move briefly and crash.
- “Ready” should reflect true competence (lap completion reliability), not short-term survival.
- User should see meaningful progress metrics that correlate with behavior quality.

### Acceptance criteria

- Within a reasonable local training budget, completion/success rate rises and remains stable (e.g. >70% over recent window on non-trivial tracks).
- Median episode length increases while collision rate decreases.
- “Ready” appears only after sustained lap completion criteria (not single lucky runs).

## Architect Agent

### Root cause in current backend

- Current loop is “PPO-ish” but not robust PPO/GAE sequence training with recurrent batching.
- Reward and termination currently overemphasize near-wall penalties without enough progress structure.
- Environment fidelity and curriculum are minimal, so policy collapses into unstable behavior.

### Quality plan (backend-first, highest impact)

1. **Implement true recurrent PPO properly**
   - Collect rollouts with hidden states.
   - Use GAE-lambda advantages and clipped PPO objective.
   - Mini-batch by sequence chunks (preserve temporal order).
   - Value loss + entropy bonus with annealing.

2. **Better observation and normalization**
   - Normalize ray distances, speed, heading error, lateral offset, and previous action.
   - Maintain running mean/std normalizer (training/inference consistency).

3. **Reward redesign for lap driving**
   - Main positive signal: forward arc-length progress along centerline.
   - Sparse big reward: valid start/finish crossing after sufficient progress.
   - Penalties: wall collision impulse, off-track dwell, steering jerk, reverse progress.
   - Avoid per-step terms that encourage oscillation.

4. **Curriculum and domain randomization**
   - Start on easier tracks (wider road, lower speed cap), then increase difficulty.
   - Randomize spawn offset/heading slightly so policy generalizes.

5. **Action smoothing + vehicle constraints**
   - Tanh action output with low-pass filtered controls in env step.
   - Clamp steering rate and acceleration to physically plausible limits.

6. **Readiness gating**
   - `ready` only when all hold over moving window:
     - success rate threshold
     - low collision rate
     - stable return (variance bound)

7. **Tooling**
   - Add checkpoint save/load every N generations.
   - Export best checkpoint by validation metric, not latest.
   - Add deterministic evaluation episodes and report in `metrics`.

### Files affected

- `apps/api/main.py`: PPO/GAE recurrent trainer, reward shaping, readiness thresholds, checkpoint manager.
- `index.html`: richer metrics display (success, collision rate, eval score, best checkpoint generation).

## Devil’s Advocate

- Full recurrent PPO is significantly more complex; risk of long implementation cycle.
  - Mitigation: staged rollout (first: true PPO+GAE feedforward baseline, then recurrent batching).
- Over-shaped rewards can create shortcut behavior.
  - Mitigation: keep progress and lap completion as dominant signals, run ablation tests.
- Local CPU constraints may limit “highest quality”.
  - Mitigation: adjustable quality presets (`fast`, `balanced`, `quality`) controlling rollout size and epochs.

## Verdict

**CONCERN** — proceed with staged quality uplift:
- Phase 1: robust PPO/GAE + reward/curriculum + readiness gating.
- Phase 2: full recurrent sequence PPO batching + checkpoints + evaluation.

---

Date: 2026-03-26
Topic: Browser console errors (app vs extension noise)

## PM Agent

### Scope / user impact

- Console currently shows noisy errors during use; user cannot trust whether the app is broken.
- We need to separate:
  - app-actionable errors
  - browser extension/environment noise
- Keep console clean enough that real training/race failures are obvious.

### Acceptance criteria

- App-generated errors are reduced to known/expected messages only.
- Non-app errors are documented so they do not block development.
- Training/race failures produce explicit app logs with clear recovery actions.

## Architect Agent

### Observed errors and likely ownership

- `Cannot set properties of undefined ... DynamicTree.js:119`
  - Very likely from a browser extension content script, not `index.html`.
- `Could not establish connection. Receiving end does not exist.`
  - Also characteristic of extension messaging, not app runtime.
- `:5173/favicon.ico 404`
  - App-owned but harmless; fix by adding favicon link or `favicon.ico`.
- Backend terminal logs show normal training API calls and WS traffic; no fatal backend exception from these console messages.

### Plan

1. **Isolate ownership first**
   - Reproduce in Incognito with extensions disabled.
   - If `DynamicTree.js` / messaging errors disappear, classify as external noise.

2. **Clean app console path**
   - Add favicon reference in `index.html` (or add `favicon.ico`) to remove 404.
   - Wrap frontend training connect/start/stop with explicit try/catch + `console.error("[wonder-car] ...")`.
   - Add WS `onerror/onclose` logs with session id and transition reason.

3. **Add defensive guards**
   - Ensure all DOM refs used in setup are null-checked before property assignment.
   - Keep idempotent `disconnectBackend()` and state transitions to avoid “undefined” writes from app code.

4. **Quality of diagnostics**
   - Prefix app logs with `[wonder-car]` so extension errors are visually distinct.
   - Optionally add a debug toggle to silence verbose per-frame logs.

## Devil’s Advocate

- If we assume extension-only too early, a real bug could be missed.
  - Mitigation: reproducibility test in clean profile is mandatory.
- Suppressing too many logs may hide useful details.
  - Mitigation: keep warnings/errors always on; silence only debug info.

## Verdict

**APPROVE** — treat `DynamicTree.js` and “receiving end does not exist” as probable extension noise pending clean-profile confirmation; fix app-owned favicon + improve explicit app logging/guards.

---

Date: 2026-03-26
Topic: UX, physics, training HUD, and practise/race visibility

## PM Agent

### Scope

- Remove **Connect backend** button; replace with implicit connect + a **compact status indicator** (not a large circular “signal” light).
- **Physics feel:** slower, smoother steering; fix **random jumping** and **post-turn acceleration slide** into walls.
- **UI:** race-themed, sleek, sharp (not generic panels).
- **Training:** higher quality (backend); **larger car** hitbox/visual.
- **Modes:** rename **Drive (random)** → **Practise**; **random AI car** visible only in **Practise** and **Race** (not Draw/Train).
- **Training controls:** single **Start/Stop training** toggle (no separate connect).
- **HUD:** show **player speed** and **ghost speed**; training HUD shows **human-readable** stats (lap/success/collision/learning progress).

### Acceptance criteria

- No standalone “Connect backend” button; training start attempts connect automatically; status is glanceable.
- Steering feels damped and rate-limited; no violent snap turns.
- No unexplained body teleport/jitter during normal drive.
- Acceleration after a turn does not instantly spike lateral loss of control without player input (within reason).
- Practise mode shows random/practise car; Train mode does not show it unless product says ghost-only (per requirement: random only practise+race).
- Training toggle works as one control.
- Speed readouts update smoothly.

## Architect Agent

### UI / chrome

- Remove `btnConnect`; add small **status pill** or **LED strip** near title (connected / training / offline) using existing pill styles, not a huge circular element.
- Single **`btnTrainToggle`**: `idle` → start training (POST session, track, WS, train/start); `running` → train/stop.
- Rename mode label **Practise** (internal id `practise` or keep `drive_random` with new label).

### Visibility rules

- **Draw / Train:** hide “practise/random” car body (player car for random drive) — clarify: user car in **Practise** is the controllable car; **Train** hides player per prior plan, shows ghost only.
- **Practise + Race:** show player + (ghost when racing); random controller only active in Practise when chosen.

### Physics (Planck)

- Increase car **fixture size** (box half-extents × ~1.25–1.4).
- **Steering:** reduce `applyTorque` gain; add **per-frame steering slew limit** (target steer from input smoothed).
- **Traction:** scale lateral friction with speed; optional **longitudinal cap** on combined force; reduce impulse spikes from `post-solve` damage.
- **Stability:** fixed timestep already present — add **sub-steps** (2–4) per frame if needed, or lower max force.
- **Jumping:** often from tunneling or huge torque — clamp angular velocity, ensure no double application of forces, verify ghost bodies inactive when not in use.

### Training quality

- Continue backend roadmap: PPO/GAE, reward shaping, curriculum (`apps/api/main.py`).

### HUD

- Speed: `speed = |v|` in display units (px/s or km/h label).
- Training HUD labels: **Status**, **Learning progress** (generation), **Best recent lap / success**, **Crashes per minute** (derived), plain language tooltips.

### Files

- `index.html` (CSS overhaul, physics constants, mode rename, toggle, visibility, HUD).
- `apps/api/main.py` (training quality iterations).

## Devil’s Advocate

- Hiding cars in wrong mode can confuse users who expect to see the player in Train.
  - Mitigation: Train stays **ghost-only** per earlier requirement; document in HUD.
- Auto-connect on every train click may spam sessions if backend flaps.
  - Mitigation: reuse `backendSessionId` until explicit end-session or failure.
- UI redesign can regress accessibility (contrast).
  - Mitigation: keep WCAG-friendly accent on text.

## Verdict

**CONCERN** — proceed with UI/physics/mode changes in `index.html` and staged training improvements in backend; auto-connect with session reuse and a compact status indicator (no large circular signal light).

---

Date: 2026-03-26
Topic: Frontend bugfix plan (game over, spawn, start line, end session, rays)

## PM Agent

### Scope / user impact

- **Game over overlay stuck:** User must be able to dismiss or recover (restart) without refreshing; overlay must not block the only recovery path.
- **Cars jump / don’t move:** Driving and ghost motion must feel consistent; no bodies at wrong scale or teleported every frame incorrectly.
- **Starting line:** User must see a clear default or guided way to set START (not only hidden keys).
- **End session:** Must visibly reset track, session, cars, overlay, and backend connection state.
- **Spawn on track:** Player (and race ghost) must spawn on/near the road centerline when a track exists.
- **Rays:** Sensor rays are **training-only** (ghost in Train mode); not drawn for the player car in Drive/Race.

### Acceptance criteria

- After GAME OVER, user can tap **Restart** or **End session** and the overlay clears; physics resumes when appropriate.
- With a drawn track, cars spawn at a defined start (centerline + start line if set, else first center point).
- **End session** clears canvas track state, storage, overlay, and disconnects/resets backend session.
- In **Train** mode only, ghost training rays render; player car has **no** ray fan in Drive/Race/Draw.

## Architect Agent

### Root causes (from current `index.html` behavior)

- **Overlay stuck:** `triggerGameOver` sets `display: grid` but there is no dedicated **Restart** / **Continue** that calls `resetWorld()` or clears `gameover`; `endSession` should work but user perception may fail if state doesn’t fully reset or if overlay z-index/pointer issues exist.
- **Random cars / bad motion:** Two `pointerdown` handlers on the same canvas (draw vs place start/finish) can conflict; ghost may be activated with backend transforms while physics still steps; mixing **Planck world coords** (meters) vs **canvas pixels** must stay consistent (`pxPerMeter`).
- **No visible start line:** Start/finish only via keys `1`/`2` + drag—undiscoverable; no default `startLine` when track is saved.
- **End session “no effect”:** Likely partial reset (e.g. backend WS still connected, or UI state not cleared) or user expects visible feedback; must explicitly: clear `localStorage`/`sessionStorage`, `resetWorld()`, close WebSocket, null `backendSessionId`.
- **Spawn outside track:** Default spawn `(3,3)` in physics space when no track; with track, use **start line midpoint** or **first centerline point** with heading along tangent.
- **Rays:** `castRays(playerBody, …)` runs every frame; gate: only draw rays for **ghost** when `mode === TRAIN` (and optionally only when backend connected).

### File changes

- **`index.html` only** for this pass (single-file prototype).

### Implementation sequence

1. Add **Restart** (and/or **Dismiss game over**) button wired to `resetWorld()` + `state.game.gameover = false` + hide overlay.
2. Unify canvas **pointer handlers**: one `pointerdown/move/up` path; branch by `mode` and optional “place start/finish” sub-mode from explicit UI buttons instead of only keys.
3. On track save / first valid polyline: **default `startLine`** to a short segment at first centerline segment (and optional finish near last segment) unless user already set them.
4. **`spawnCarsOnTrack()`:** compute position + angle from start line or centerline[0]→[1]; call after `rebuildWalls()` and when entering Race/Drive/Train.
5. **`endSession()`:** close WS, clear backend state, clear `simBuffer`, then storage + `resetWorld()` + visual HUD update.
6. **Rays:** remove player `castRays`; keep `drawBackendRays` / ghost `castRays` only when `mode === TRAIN`.

## Devil’s Advocate

- Auto-generated start/finish might point the wrong way on self-intersecting tracks—mitigate with “flip heading” button later.
- Closing WS on end session must not throw; use try/catch.
- Ensure `resetWorld` doesn’t double-create bodies without destroying old world (already pattern: new `pl.World`).

## Verdict

**APPROVE** — implement the above in `index.html` with explicit UX for restart and start line, and gate ray rendering to training ghost only.

---

Date: 2026-03-26
Topic: Practise flicker / spin, random crashes, and two console errors

## PM Agent

### Scope

- **Practise:** stop “multiple cars” flicker and endless **spinning in circles**; reduce **random crashing** during practise.
- **Console:** explain **`Receiving end does not exist`** and **`DynamicTree.js` `height`** errors in plain language.

### Acceptance criteria

- In Practise, user sees **one** stable car (no strobing / stacked ghosts).
- Driving is predictable: either **manual (arrows)** by default or an explicit **“Auto demo”** toggle for sinusoidal AI—no hidden forced chaos.
- Extension-related console noise is documented; app errors remain prefixed with `[wonder-car]`.

## Architect Agent

### Likely app causes (practise)

1. **“Multiple cars” / flicker**
   - Visual: blast particles + car + HUD refresh can look like duplicates; verify only one `renderCar` path per body.
   - **Teleport every mode change:** `setMode()` calls `spawnCarsOnTrack()` whenever mode switches—rapid toggles cause jumpy motion.
   - **Unhandled promise:** `uploadTrackToBackend()` on pointer-up may **reject** if backend is down → console noise and odd state; wrap in `try/catch`.

2. **Rotating in circles**
   - Practise currently uses **sin/cos random steering** (`performance.now()`), which naturally produces **orbiting** behavior—not a rendering bug.
   - **Fix:** default Practise to **player-only arrows**; move random AI to optional **“Auto (demo)”** or reduce amplitude/frequency.

3. **Random crashing**
   - Same as above: aggressive random steer + walls.
   - Add **traction** tuning already planned; optionally **cap** random steer magnitude.

### Console errors (usually not wonder-car)

- **`Uncaught (in promise) Error: Could not establish connection. Receiving end does not exist.`**
  - Typical **browser extension** (`chrome.runtime.sendMessage` / messaging) when the extension background page is not connected. **Not from `index.html`** unless the app used extension APIs (it does not).

- **`DynamicTree.js:119 Cannot set properties of undefined (setting 'height')`**
  - **`DynamicTree.js` is not part of this repo**—injected by an extension or another script on the page. Planck uses internal structures but not a file named `DynamicTree.js` in our bundle.

**Mitigation:** reproduce in **Incognito with extensions disabled**; if errors vanish, treat as external.

### Implementation sequence

1. Add **`try/catch`** around `uploadTrackToBackend()` from UI events; log with `[wonder-car]` only.
2. **Debounce / guard** `spawnCarsOnTrack()`—only when mode **changes** or track **geometry** changes, not redundant `setMode` calls with same mode.
3. Practise: **manual drive default**; optional **Auto demo** button for sinusoidal control with softer gains.
4. Optional: clear **blast particles** when entering Practise from Train.

## Devil’s Advocate

- Removing random AI entirely may disappoint users who liked “chaos demo.”
  - Mitigation: keep as **opt-in**.
- Debouncing spawn wrong can leave car off-track after edits.
  - Mitigation: always respawn after `commitStroke()` and line placement (already).

## Verdict

**CONCERN** — treat console errors as **extension-noise** until reproduced in a clean profile; fix Practise behavior by **splitting manual vs auto demo** and **hardening async track upload** + spawn guards.

---

## `/develop` completion — Practise fix (2026-03-26)

**Senior Dev Reviewer:** `setMode` uses `prev !== next` before `spawnCarsOnTrack`; `stepPlayer` applies demo AI only when `mode === PRACTISE && state.practise.autoDemo`; Race remains arrow-only. Matches `patterns.md`.

**Security & Perf:** No new DOM sinks; `uploadTrackToBackend` catches network errors only.

**Integration:** `updatePractiseUi()` syncs checkbox disabled state; leaving Practise clears auto-demo. Track edits still call `spawnCarsOnTrack` where needed.

**QA:** No Vitest in repo; manual verification: Practise default manual, optional auto demo, no spawn flicker on double-clicking same mode tab.

**Outcome:** Implemented in `index.html`; `progress.md` / `patterns.md` updated.

---

## `/develop` completion — car not moving + training quality (2026-03-26)

**Senior Dev Reviewer:** Frontend guards and focus match `patterns.md`. Backend returns segmented on lap; REINFORCE uses same `adv` across PPO epochs; empty-buffer guard.

**Security & Perf:** No new attack surface; training math unchanged order of magnitude.

**Integration:** Train mode unchanged (player inactive); Practise/Race unchanged.

**QA:** `py_compile` on `main.py`; manual smoke recommended.

---

## `/plan` — Car not moving + training quality (2026-03-26)

### PM Agent

- **Scope:** (1) Player car appears not to move. (2) Training quality still poor.
- **Acceptance:** Driving works in **Practise/Race** with clear UX; training improves via **correct RL targets** (returns / advantage), not only reward shaping.

### Architect Agent

- **Car:** `stepPlayer` is gated by **mode** (not Train) and **gameover**; physics loop also skips when gameover. **Train** intentionally **deactivates** the player — likely user confusion. Add **guards** on `playerBody`, **HUD** clarity, optional **canvas focus**.
- **Training:** Value net is trained against **per-step reward** instead of **returns**; “PPO” is effectively **behavior cloning** on sampled actions. Fix **discounted returns**, **advantage = R − V**, then **policy gradient** or consistent supervised step on **advantage-weighted** errors.

### Devil’s Advocate

- Do not add **idle throttle** without confirming it does not break “parked” feel.
- **Return** math must handle **early episode end** and **finish-line `continue`** branches.
- **Sim mismatch** remains a ceiling until Planck-aligned training or in-browser RL.

### Verdict

**CONCERN** — **APPROVE** split **frontend** / **backend** plan recorded in **`.cursor/memory/progress.md`** (addendum **Player car “not moving” + poor training quality**).

---

## `/plan` — Backend pill “always offline” (2026-03-26)

### PM Agent

- **Scope:** Users report the backend pill **always shows offline** while developing with the API running.
- **Acceptance:** Distinguish **API unreachable** vs **API up but no training session**; **WebSocket URL** must match `baseUrl`; optional **accurate** `connected` only after WS `open`.
- **Impact:** Less confusion during local dev; clearer when to start uvicorn vs when training failed.

### Architect Agent

- **Root cause:** `updateBackendPill()` treats **offline** as `!connected && !backendSessionId`. Nothing sets **`connected`** until **Start training** opens a session + WS — so the UI stays **offline** even when FastAPI is healthy.
- **Secondary bug:** WebSocket uses **hardcoded** `ws://localhost:8000/...` while REST uses **`state.backend.baseUrl`** — port/host drift breaks training.
- **Fix:** `GET /health` + periodic **`fetch`** → **`apiReachable`**; extend pill states; **derive WS URL** from `baseUrl`.

### Devil’s Advocate

- Health-only “online” can mislead if **WS** is blocked; training flow still the source of truth.
- **CORS** must include every dev origin; **interval** polling should be modest.
- **Precedence** in `updateBackendPill` must be tested so **training** still wins over **idle**.

### Verdict

**CONCERN** — Approved with mitigations above. Plan written to **`.cursor/memory/progress.md`** (addendum **Backend pill “always offline”**).
