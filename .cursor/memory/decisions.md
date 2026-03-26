# Architecture decisions — wonder-car

## Product

- **Name:** wonder-car.
- **Experience:** User-drawn track, explicit start/finish, RL agent learns navigation, then **two-car race** (learned vs another).
- **Constraints:** Local-only initially; no auth; no server database.

## Stack

- **Language:** TypeScript.
- **UI:** React + Vite (monorepo).
- **Data:** None required for v0; optional later: localStorage or file export for tracks/models.

## ML / RL

- **Recurrent RL** implies state that benefits from memory (e.g. partial observability or sequence modeling). Implementation detail TBD: custom env + algorithm in TS, or WASM-bound library, or worker-isolated training loop.
- **API style:** Prefer the simplest transport that fits: in-process for single-tab training; add **WebSocket** only if we need live metric streams or future head-to-head sync.

## Open questions

- Exact RL algorithm and recurrence placement (network vs environment).
- How “another car” is defined (rule-based opponent, second trained policy, human control).
- Whether tracks and policies are persisted (format and location).
