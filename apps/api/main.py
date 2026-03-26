from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n <= 1e-9:
        return (0.0, 0.0)
    return (vx / n, vy / n)


def _seg_intersect_ray(
    ox: float, oy: float, dx: float, dy: float, ax: float, ay: float, bx: float, by: float
) -> Optional[float]:
    # Ray: o + t*d, t>=0
    # Segment: a + u*(b-a), u in [0,1]
    rx, ry = dx, dy
    sx, sy = (bx - ax), (by - ay)
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-9:
        return None
    qpx, qpy = (ax - ox), (ay - oy)
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def offset_polyline(points: List[Tuple[float, float]], offset: float) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return points[:]
    out: List[Tuple[float, float]] = []
    n = len(points)
    for i in range(n):
        x, y = points[i]
        if i == 0:
            x2, y2 = points[i + 1]
            tx, ty = _unit(x2 - x, y2 - y)
        elif i == n - 1:
            x1, y1 = points[i - 1]
            tx, ty = _unit(x - x1, y - y1)
        else:
            x1, y1 = points[i - 1]
            x2, y2 = points[i + 1]
            t1x, t1y = _unit(x - x1, y - y1)
            t2x, t2y = _unit(x2 - x, y2 - y)
            tx, ty = _unit(t1x + t2x, t1y + t2y)
            if abs(tx) < 1e-9 and abs(ty) < 1e-9:
                tx, ty = t2x, t2y
        # left normal
        nx, ny = (-ty, tx)
        out.append((x + nx * offset, y + ny * offset))
    return out


def polyline_segments(points: List[Tuple[float, float]]) -> List[Tuple[float, float, float, float]]:
    segs: List[Tuple[float, float, float, float]] = []
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        segs.append((ax, ay, bx, by))
    return segs


def _centerline_total_length(center: List[Tuple[float, float]]) -> float:
    if len(center) < 2:
        return 0.0
    s = 0.0
    for i in range(len(center) - 1):
        ax, ay = center[i]
        bx, by = center[i + 1]
        s += math.hypot(bx - ax, by - ay)
    return s


def _centerline_project(center: List[Tuple[float, float]], px: float, py: float) -> Tuple[float, float, float]:
    """
    Closest point on polyline to (px, py): returns unit tangent (tx, ty) at that point
    and arc-length `s` from center[0] along the polyline to the foot.
    """
    if len(center) < 2:
        return (1.0, 0.0, 0.0)
    best_d2: Optional[float] = None
    best_tx, best_ty = 1.0, 0.0
    best_s = 0.0
    cum_start = 0.0
    for i in range(len(center) - 1):
        ax, ay = center[i]
        bx, by = center[i + 1]
        abx, aby = bx - ax, by - ay
        seg_len2 = abx * abx + aby * aby
        if seg_len2 < 1e-12:
            cum_start += 0.0
            continue
        seg_len = math.sqrt(seg_len2)
        axpx, aypy = px - ax, py - ay
        u = _clamp((axpx * abx + aypy * aby) / seg_len2, 0.0, 1.0)
        cx = ax + u * abx
        cy = ay + u * aby
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_tx, best_ty = _unit(abx, aby)
            best_s = cum_start + u * seg_len
        cum_start += seg_len
    return (best_tx, best_ty, best_s)


def _segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> bool:
    def ccw(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def _discounted_returns(rewards: List[float], gamma: float, segment_starts: List[int]) -> np.ndarray:
    """Monte Carlo returns per segment (resets on lap finish split segments)."""
    n = len(rewards)
    R = np.zeros(n, dtype=np.float64)
    if n == 0:
        return R
    r = np.array(rewards, dtype=np.float64)
    ss = sorted(set(segment_starts))
    if not ss or ss[0] != 0:
        ss = [0] + [x for x in ss if x > 0]
    ss = ss + [n]
    for seg in range(len(ss) - 1):
        s, e = ss[seg], ss[seg + 1]
        if s >= e:
            continue
        G = 0.0
        for t in range(e - 1, s - 1, -1):
            G = r[t] + gamma * G
            R[t] = G
    return R


def raycast_to_segments(
    origin: Tuple[float, float], direction: Tuple[float, float], segs: List[Tuple[float, float, float, float]], max_dist: float
) -> float:
    ox, oy = origin
    dx, dy = direction
    best = None
    for ax, ay, bx, by in segs:
        t = _seg_intersect_ray(ox, oy, dx, dy, ax, ay, bx, by)
        if t is None:
            continue
        if t < 0:
            continue
        if best is None or t < best:
            best = t
    if best is None:
        return max_dist
    return min(best, max_dist)


class GRUPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.pi = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 2))
        self.v = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.tensor([-0.3, -0.5], dtype=torch.float32))

    def forward(
        self, obs_seq: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_seq: [B,T,obs_dim]
        out, hn = self.gru(obs_seq, h)
        last = out[:, -1, :]
        mu = self.pi(last)
        v = self.v(last).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std, v, hn

    @torch.no_grad()
    def act(self, obs: np.ndarray, h: Optional[torch.Tensor]) -> Tuple[np.ndarray, torch.Tensor]:
        obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, 1, -1)
        mu, log_std, _v, hn = self.forward(obs_t, h)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        a = mu + eps * std
        a = torch.tanh(a)
        return a.squeeze(0).cpu().numpy(), hn


@dataclass
class Track:
    center: List[Tuple[float, float]] = field(default_factory=list)
    road_half_width: float = 18.0
    start: Optional[List[Tuple[float, float]]] = None
    finish: Optional[List[Tuple[float, float]]] = None
    left: List[Tuple[float, float]] = field(default_factory=list)
    right: List[Tuple[float, float]] = field(default_factory=list)
    wall_segs: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def rebuild(self) -> None:
        if len(self.center) < 2:
            self.left = []
            self.right = []
            self.wall_segs = []
            return
        self.left = offset_polyline(self.center, +self.road_half_width)
        self.right = offset_polyline(self.center, -self.road_half_width)
        self.wall_segs = polyline_segments(self.left) + polyline_segments(self.right)


@dataclass
class TrainState:
    status: str = "stopped"  # stopped | in-training | ready
    generation: int = 0
    epoch: int = 0
    timesteps: int = 0
    episode: int = 0
    avg_return: float = 0.0
    success_rate: float = 0.0
    ready: bool = False


@dataclass
class Session:
    id: str
    created_at_ms: int
    track: Track = field(default_factory=Track)
    train: TrainState = field(default_factory=TrainState)
    task: Optional[asyncio.Task] = None
    stop_flag: bool = False


SESSIONS: Dict[str, Session] = {}


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173", "http://127.0.0.1:5173", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/session")
async def create_session() -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    sess = Session(id=sid, created_at_ms=_now_ms())
    SESSIONS[sid] = sess
    return {"sessionId": sid}


@app.post("/session/{session_id}/track")
async def upload_track(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"ok": False, "error": "unknown_session"}
    center = payload.get("centerline", [])
    road_width = float(payload.get("roadWidth", 36.0))
    start = payload.get("startLine")
    finish = payload.get("finishLine")
    sess.track.center = [(float(p[0]), float(p[1])) for p in center]
    sess.track.road_half_width = road_width / 2.0
    sess.track.start = start
    sess.track.finish = finish
    sess.track.rebuild()
    return {"ok": True}


@app.post("/session/{session_id}/train/start")
async def train_start(session_id: str) -> Dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"ok": False, "error": "unknown_session"}
    if sess.task and not sess.task.done():
        return {"ok": True, "alreadyRunning": True}
    sess.stop_flag = False
    sess.train = TrainState(status="in-training")
    sess.task = asyncio.create_task(_train_loop(sess))
    return {"ok": True}


@app.post("/session/{session_id}/train/stop")
async def train_stop(session_id: str) -> Dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"ok": False, "error": "unknown_session"}
    sess.stop_flag = True
    if sess.task:
        try:
            await asyncio.wait_for(sess.task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
    sess.train.status = "stopped"
    return {"ok": True}


@app.post("/session/{session_id}/train/reset")
async def train_reset(session_id: str) -> Dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"ok": False, "error": "unknown_session"}
    sess.stop_flag = True
    if sess.task and not sess.task.done():
        try:
            await asyncio.wait_for(sess.task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
    sess.task = None
    sess.train = TrainState(status="stopped")
    return {"ok": True}


class Hub:
    def __init__(self) -> None:
        self.clients: Dict[str, List[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def add(self, session_id: str, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.setdefault(session_id, []).append(ws)

    async def remove(self, session_id: str, ws: WebSocket) -> None:
        async with self.lock:
            arr = self.clients.get(session_id)
            if not arr:
                return
            try:
                arr.remove(ws)
            except ValueError:
                return
            if not arr:
                self.clients.pop(session_id, None)

    async def broadcast(self, session_id: str, msg: Dict[str, Any]) -> None:
        payload = json.dumps(msg, separators=(",", ":"))
        async with self.lock:
            arr = list(self.clients.get(session_id, []))
        if not arr:
            return
        dead: List[WebSocket] = []
        for ws in arr:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self.lock:
                arr2 = self.clients.get(session_id, [])
                for ws in dead:
                    if ws in arr2:
                        arr2.remove(ws)


HUB = Hub()


@app.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    await HUB.add(session_id, ws)
    try:
        # Immediately send a hello + current status.
        sess = SESSIONS.get(session_id)
        if sess is not None:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "metrics",
                        "ts": _now_ms(),
                        "status": sess.train.status,
                        "generation": sess.train.generation,
                        "epoch": sess.train.epoch,
                        "timesteps": sess.train.timesteps,
                        "episode": sess.train.episode,
                        "avgReturn": sess.train.avg_return,
                        "successRate": sess.train.success_rate,
                    }
                )
            )
        while True:
            # Keep connection open; we don't require client messages for MVP.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await HUB.remove(session_id, ws)


async def _train_loop(sess: Session) -> None:
    """
    MVP training loop:
    - Uses true GRU policy (RNN).
    - Runs a simple kinematic sim (not full physics) so it stays fast.
    - Reward: arc-length progress along centerline + optional finish crossing; tangent from closest polyline point.
    - Streams live sim_state + metrics with epoch/generation semantics.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    sensor_n = 11
    max_ray = 120.0
    obs_dim = sensor_n + 4  # rays + speed + heading_err + health + progress_along [0,1]
    policy = GRUPolicy(obs_dim=obs_dim, hidden_dim=64)
    optim = torch.optim.Adam(policy.parameters(), lr=2.5e-4)
    gamma = 0.99

    ppo_epochs = 3
    rollout_steps = 320

    # Success if a lap was completed this rollout, or survived with few wall hits.
    ready_threshold = 0.55

    # Car state in "track space" (same coordinate system as frontend canvas).
    x, y = 320.0, 180.0
    heading = 0.0
    speed = 0.0
    health = 100.0

    # RNN hidden state
    h: Optional[torch.Tensor] = None

    # Rolling metrics
    returns_window: List[float] = []
    success_window: List[float] = []

    sess.train.status = "in-training"
    sess.train.ready = False

    last_metrics_emit = 0.0
    last_sim_emit = 0.0

    def reset_episode() -> None:
        nonlocal x, y, heading, speed, health, h
        sess.train.episode += 1
        if len(sess.track.center) >= 2:
            x, y = sess.track.center[0]
            x1, y1 = sess.track.center[1]
            heading = math.atan2(y1 - y, x1 - x)
        elif sess.track.center:
            x, y = sess.track.center[0]
            heading = 0.0
        else:
            x, y = 320.0, 180.0
            heading = 0.0
        speed = 0.0
        health = 100.0
        h = None

    reset_episode()

    # Simple buffers for PPO-ish loss (very lightweight, not production PPO).
    while not sess.stop_flag:
        track_len = max(_centerline_total_length(sess.track.center), 1e-6)
        sess.train.generation += 1
        gen_return = 0.0
        wall_hits = 0
        lap_finished = False

        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []
        h_buf = []
        seg_starts: List[int] = [0]

        for t in range(rollout_steps):
            sess.train.timesteps += 1

            px, py = x, y
            tx, ty, s_old = _centerline_project(sess.track.center, x, y)
            prog = _clamp(s_old / track_len, 0.0, 1.0)

            # sensors
            rays = []
            if sess.track.wall_segs:
                for i in range(sensor_n):
                    a = (i / (sensor_n - 1)) * math.radians(140) - math.radians(70)
                    ang = heading + a
                    dx, dy = math.cos(ang), math.sin(ang)
                    dist = raycast_to_segments((x, y), (dx, dy), sess.track.wall_segs, max_ray)
                    rays.append(dist / max_ray)
            else:
                rays = [1.0] * sensor_n

            # heading error: dot/cross with tangent
            fx, fy = math.cos(heading), math.sin(heading)
            dot = _clamp(fx * tx + fy * ty, -1.0, 1.0)
            cross = fx * ty - fy * tx
            heading_err = math.atan2(cross, dot) / math.pi  # [-1,1]

            obs = np.array(rays + [speed / 8.0, heading_err, health / 100.0, prog], dtype=np.float32)

            obs_t = torch.from_numpy(obs).view(1, 1, -1)
            mu, log_std, v, hn = policy.forward(obs_t, h)
            std = torch.exp(log_std)
            eps = torch.randn_like(mu)
            a = torch.tanh(mu + eps * std)
            throttle = float(a[0, 0].item())
            steer = float(a[0, 1].item())
            h = hn

            # Kinematic sim step
            dt = 1.0 / 60.0
            speed = _clamp(speed + throttle * 12.0 * dt - 0.8 * speed * dt, 0.0, 10.0)
            heading += steer * 1.55 * dt
            x += math.cos(heading) * speed * 24.0 * dt
            y += math.sin(heading) * speed * 24.0 * dt

            # Wall "collision" approximation: penalize if rays are very small.
            near = min(rays) if rays else 1.0
            hit = near < 0.08
            if hit:
                wall_hits += 1
                dmg = (0.08 - near) * 150.0
                health = max(0.0, health - dmg)

            _, _, s_new = _centerline_project(sess.track.center, x, y)
            ds = max(0.0, s_new - s_old)
            backward = max(0.0, s_old - s_new)

            # Reward: forward alignment + arc-length progress + light speed / steer shaping
            forward = fx * tx + fy * ty
            r = (
                0.2 * forward
                + 0.18 * (ds / max(track_len * 0.01, 1.0))
                + 0.05 * (speed / 10.0)
                - 0.08 * backward
                - (0.55 if hit else 0.0)
                - 0.035 * abs(steer)
            )
            if health <= 0:
                r -= 1.0

            fin = sess.track.finish
            if fin and len(fin) == 2:
                f0 = (float(fin[0][0]), float(fin[0][1]))
                f1 = (float(fin[1][0]), float(fin[1][1]))
                if _segments_intersect((px, py), (x, y), f0, f1):
                    r += 10.0
                    lap_finished = True
                    gen_return += r
                    obs_buf.append(obs)
                    act_buf.append([throttle, steer])
                    rew_buf.append(r)
                    val_buf.append(float(v.item()))
                    done_buf.append(1.0 if health <= 0 else 0.0)
                    h_buf.append(None)
                    seg_starts.append(len(obs_buf))
                    reset_episode()
                    continue

            gen_return += r

            obs_buf.append(obs)
            act_buf.append([throttle, steer])
            rew_buf.append(r)
            val_buf.append(float(v.item()))
            done_buf.append(1.0 if health <= 0 else 0.0)
            h_buf.append(None)  # placeholder; we don't train on hidden state in this MVP

            # Emit sim_state ~15Hz
            now = time.time()
            if now - last_sim_emit >= (1.0 / 15.0):
                last_sim_emit = now
                await HUB.broadcast(
                    sess.id,
                    {
                        "type": "sim_state",
                        "ts": _now_ms(),
                        "episode": sess.train.episode,
                        "x": x,
                        "y": y,
                        "heading": heading,
                        "speed": speed,
                        "health": health,
                        "rays": rays,
                        "maxRay": max_ray,
                        "sensorFovDeg": 140,
                    },
                )

            if health <= 0:
                break

        # Compute readiness proxy
        success = (
            1.0
            if lap_finished or (wall_hits < max(1, rollout_steps // 12) and health > 0)
            else 0.0
        )
        returns_window.append(gen_return)
        success_window.append(success)
        returns_window = returns_window[-40:]
        success_window = success_window[-40:]
        sess.train.avg_return = float(np.mean(returns_window)) if returns_window else gen_return
        sess.train.success_rate = float(np.mean(success_window)) if success_window else success

        if len(obs_buf) == 0:
            if health <= 0:
                reset_episode()
            await asyncio.sleep(0)
            continue

        ret_np = _discounted_returns(rew_buf, gamma, seg_starts)
        ret_arr = torch.tensor(ret_np, dtype=torch.float32)
        obs_arr = torch.tensor(np.array(obs_buf, dtype=np.float32))
        act_arr = torch.tensor(np.array(act_buf, dtype=np.float32))
        val_arr = torch.tensor(np.array(val_buf, dtype=np.float32))
        adv = ret_arr - val_arr.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)

        for e in range(1, ppo_epochs + 1):
            if sess.stop_flag:
                break
            sess.train.epoch = e

            # Recompute distribution for each step as independent (sequence training is a bigger lift).
            mu_list = []
            logstd_list = []
            v_list = []
            htmp: Optional[torch.Tensor] = None
            for i in range(obs_arr.shape[0]):
                mu, log_std, v, htmp = policy.forward(obs_arr[i].view(1, 1, -1), htmp)
                mu_list.append(mu.squeeze(0))
                logstd_list.append(log_std.squeeze(0))
                v_list.append(v.squeeze(0))
            mu_t = torch.stack(mu_list, dim=0)
            logstd_t = torch.stack(logstd_list, dim=0)
            std_t = torch.exp(logstd_t)

            # REINFORCE-style: Gaussian log-prob in pre-tanh space; actions stored are tanh(z).
            z = torch.atanh(torch.clamp(act_arr, -0.999999, 0.999999))
            logp_dim = -0.5 * ((z - mu_t) ** 2) / (std_t**2 + 1e-8) - torch.log(std_t + 1e-8) - 0.5 * math.log(2 * math.pi)
            logp = logp_dim.sum(dim=-1)
            policy_loss = -(adv * logp).mean()
            value_loss = ((torch.stack(v_list) - ret_arr) ** 2).mean() * 0.25
            entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std_t)).mean()
            loss = policy_loss + value_loss - 0.01 * entropy

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            # Emit metrics ~3Hz
            now = time.time()
            if now - last_metrics_emit >= (1.0 / 3.0):
                last_metrics_emit = now
                await HUB.broadcast(
                    sess.id,
                    {
                        "type": "metrics",
                        "ts": _now_ms(),
                        "status": sess.train.status,
                        "generation": sess.train.generation,
                        "epoch": sess.train.epoch,
                        "epoch_total": ppo_epochs,
                        "timesteps": sess.train.timesteps,
                        "episode": sess.train.episode,
                        "avgReturn": sess.train.avg_return,
                        "successRate": sess.train.success_rate,
                    },
                )

        sess.train.epoch = 0

        if sess.train.success_rate >= ready_threshold and sess.train.generation >= 8:
            sess.train.status = "ready"
            sess.train.ready = True
        else:
            sess.train.status = "in-training"

        if health <= 0:
            reset_episode()

        await asyncio.sleep(0)  # allow IO

    sess.train.status = "stopped"

