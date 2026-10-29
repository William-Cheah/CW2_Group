"""
bloodchain_env.py  -  BloodChain-v0  (owner: Member C)
=======================================================
Custom Gymnasium environment for the CW2 group project:
"An Autonomous Agentic Cold-Chain Dispatcher for Blood Products during the
Kelantan Monsoon Floods".

THE ENVIRONMENT IS SYNTHETIC. The map, distances and probabilities below are
invented for the simulation (loosely inspired by Kelantan geography). Never
present numbers produced here as real Kelantan data.

All settings follow CONTRACT.md (Section 6 of the group brief). If you change
a setting here, update CONTRACT.md and tell the group first.

One step = 15 minutes. One episode = 96 steps = 24 hours.

Plug-in points ("hooks") for the other members
----------------------------------------------
Member A replaces these attributes on env.unwrapped (see A_uncertainty/README.md):
    sensor_hook   (readings, rng) -> (readings, sensor_fault)      Sensor-noise wrapper
    belief        object with .reset(n_roads) and .update(readings)  Belief Estimator
                  -> (p_passable array, var array)
    safety        object with .decide(logger_temp_c, minutes_out)  Fuzzy Safety Supervisor
                  -> "use" | "quarantine" | "discard"
Member B does not change the environment. The Fallback Controller sits
between the policy and env.step(), and B reads the JSONL logs (logger.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# 1. Static map (CONTRACT 6.2): 1 blood bank + 4 hospitals, 7 roads
# ---------------------------------------------------------------------------
NODES = ["Blood Bank (Kota Bharu)", "Pasir Mas", "Tanah Merah", "Machang", "Kuala Krai"]
HOSPITALS = [1, 2, 3, 4]                       # node ids of hospitals
DEPOT = 0
# (u, v, travel_minutes, flood_susceptibility 0..1)
ROADS = [
    (0, 1, 30, 0.50),   # R0 Bank - Pasir Mas        (river side)
    (0, 3, 60, 0.20),   # R1 Bank - Machang          (higher ground)
    (1, 2, 45, 0.60),   # R2 Pasir Mas - Tanah Merah
    (0, 2, 75, 0.40),   # R3 Bank - Tanah Merah      (direct)
    (3, 2, 45, 0.30),   # R4 Machang - Tanah Merah
    (3, 4, 60, 0.70),   # R5 Machang - Kuala Krai
    (2, 4, 60, 0.85),   # R6 Tanah Merah - Kuala Krai (most flood-prone)
]
N_ROADS = len(ROADS)
N_VANS = 2
VAN_CAPACITY = 6                                # blood units per cold box
STEP_MIN = 15
EPISODE_STEPS = 96

# ---------------------------------------------------------------------------
# 2. Flood + demand + cold-chain parameters
# ---------------------------------------------------------------------------
# Rainfall level: 0 none, 1 light, 2 heavy, 3 extreme (hidden Markov chain, changes hourly)
SCENARIOS = {
    #                rain level start probs        P(level goes up/down per hour)
    "moderate": dict(rain_p0=[0.3, 0.4, 0.25, 0.05], up=0.15, down=0.25),   # TRAINING
    "severe":   dict(rain_p0=[0.0, 0.2, 0.4, 0.4],   up=0.30, down=0.10),   # UNSEEN TEST
}
FLOOD_P = np.array([0.0, 0.02, 0.08, 0.20])     # P(road floods this step) x susceptibility
RECEDE_P = np.array([0.30, 0.15, 0.05, 0.01])   # P(flooded road reopens this step)
REPORT_P = 0.5                                   # P(a road gets a field report this step)
REPORT_ACC = 0.9                                 # P(field report is correct)

REQ_RATE = 0.04                                  # requests per hospital per step (Poisson)
URGENT_P = 0.3
DEADLINE_STEPS = {True: 8, False: 24}            # urgent 2 h, routine 6 h
EXPIRE_AFTER = 16                                # request cancelled 4 h after deadline

TEMP_START = 4.0                                 # cold box packed at +4 C
TEMP_RISE = (0.20, 0.60)                         # C per step; each cold box differs (packing, ice, sun)
TEMP_LIMIT = 10.0                                # transport limit (cite WHO in the report)

# Reward terms (CONTRACT 6.2, C finalises) - per event
R = dict(
    on_time_urgent=10.0, on_time_routine=4.0,
    late_urgent=-3.0, late_routine=-1.0,
    unit_wasted=-5.0, temp_breach=-10.0, unsafe_use=-20.0,
    expired_urgent=-8.0, expired_routine=-3.0,
    failed_trip=-2.0,
    waiting_urgent_per_step=-0.1,   # stops the "wait forever" reward hack
)


@dataclass
class Request:
    rid: int
    hospital: int
    units: int
    urgent: bool
    created: int
    deadline: int


@dataclass
class Van:
    vid: int
    node: int = DEPOT
    busy: bool = False
    path: list = field(default_factory=list)   # remaining nodes to visit
    edge_left: int = 0                         # steps left on current edge
    target: int = -1
    cargo: list = field(default_factory=list)  # Request objects on board
    temp: float = TEMP_START                   # TRUE cold-box temperature
    rise: float = 0.4                          # TRUE warming rate of this box (hidden)
    minutes_out: int = 0                       # time since packed
    returning: bool = False


# ---------------------------------------------------------------------------
# 3. Default plug-ins (simple placeholders - Member A replaces them)
# ---------------------------------------------------------------------------
class NaiveBelief:
    """Placeholder Belief Estimator: trusts the latest report, decays to 0.5."""

    def reset(self, n_roads):
        self.p = np.full(n_roads, 0.8)

    def update(self, readings):
        rep = readings["road_reports"]          # 1 passable, 0 flooded, -1 no report
        self.p = np.where(rep >= 0, rep.astype(float), 0.9 * self.p + 0.1 * 0.5)
        return self.p.copy(), self.p * (1 - self.p)


class CrispSafety:
    """Placeholder safety rule: hard thresholds (Member A replaces with fuzzy rules)."""

    def decide(self, logger_temp_c, minutes_out):
        if logger_temp_c > TEMP_LIMIT:
            return "discard"
        if logger_temp_c > 8.0 and minutes_out > 240:
            return "quarantine"
        return "use"


def no_sensor_noise(readings, rng):
    return readings, False


# ---------------------------------------------------------------------------
# 4. The environment
# ---------------------------------------------------------------------------
class BloodChainEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scenario: str = "moderate", render_mode=None):
        super().__init__()
        assert scenario in SCENARIOS, f"scenario must be one of {list(SCENARIOS)}"
        self.scenario = scenario
        self.render_mode = render_mode
        # action 0 = wait; action 1 + v*4 + k = send van v to HOSPITALS[k]
        self.n_actions = 1 + N_VANS * len(HOSPITALS)
        self.action_space = spaces.Discrete(self.n_actions)
        self.obs_dim = N_ROADS + 3 + 4 * N_VANS + 4 * len(HOSPITALS)   # = 34
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        # hooks (Member A)
        self.sensor_hook = no_sensor_noise
        self.belief = NaiveBelief()
        self.safety = CrispSafety()
        self.graph = nx.Graph()
        for i, (u, v, mins, _) in enumerate(ROADS):
            self.graph.add_edge(u, v, rid=i, steps=max(1, round(mins / STEP_MIN)))

    # ---------------- helpers ----------------
    def _believed_graph(self):
        """Graph with only roads the agent BELIEVES are passable (p >= 0.5)."""
        g = nx.Graph()
        g.add_nodes_from(range(len(NODES)))
        for u, v, d in self.graph.edges(data=True):
            if self.p_pass[d["rid"]] >= 0.5:
                g.add_edge(u, v, **d)
        return g

    def _path(self, src, dst):
        try:
            return nx.shortest_path(self._believed_graph(), src, dst, weight="steps")
        except nx.NetworkXNoPath:
            return None

    def _path_steps(self, path):
        return sum(self.graph[a][b]["steps"] for a, b in zip(path, path[1:]))

    def _pending(self, h):
        return [q for q in self.requests if q.hospital == h]

    def action_masks(self) -> np.ndarray:
        """True = allowed. Used by MaskablePPO. Wait is always allowed."""
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True
        for v, van in enumerate(self.vans):
            if van.busy:
                continue
            for k, h in enumerate(HOSPITALS):
                if self._pending(h) and self._path(DEPOT, h) is not None:
                    mask[1 + v * len(HOSPITALS) + k] = True
        return mask

    # ---------------- sensing ----------------
    def _sense(self):
        """Sensor Layer: build raw readings, pass through A's noise hook, update belief."""
        reports = np.full(N_ROADS, -1)
        for i in range(N_ROADS):
            if self.np_random.random() < REPORT_P:
                truth = 0 if self.flooded[i] else 1
                reports[i] = truth if self.np_random.random() < REPORT_ACC else 1 - truth
        for rid in self.van_seen:                       # driver confirmed the flood (feedback loop)
            reports[rid] = 0
        readings = {
            "van_seen_flooded": sorted(self.van_seen),
            "rain_level": int(self.rain),
            "river_level": float(np.clip(0.25 * self.rain + self.np_random.normal(0, 0.05), 0, 1)),
            "road_reports": reports,
            "box_temp": np.array([v.temp + self.np_random.normal(0, 0.3) for v in self.vans]),
        }
        self.van_seen = set()
        readings, self.sensor_fault = self.sensor_hook(readings, self.np_random)
        self.readings = readings
        self.p_pass, self.p_var = self.belief.update(readings)
        self.p_pass = np.clip(np.asarray(self.p_pass, float), 0, 1)

    def _obs(self):
        o = np.zeros(self.obs_dim, np.float32)
        o[:N_ROADS] = self.p_pass
        i = N_ROADS
        o[i] = self.readings["rain_level"] / 3
        o[i + 1] = self.readings["river_level"]
        o[i + 2] = self.t / EPISODE_STEPS
        i += 3
        for v, van in enumerate(self.vans):
            # steps until the van is free again (approximate: ignores rain slow-down)
            rest = van.path if van.edge_left > 0 else [van.node] + van.path
            left = van.edge_left + (self._path_steps(rest) if len(rest) > 1 else 0)
            o[i:i + 4] = [float(not van.busy),
                          min(left, 16) / 16,
                          sum(q.units for q in van.cargo) / VAN_CAPACITY,
                          (self.readings["box_temp"][v] - 2) / 10]
            i += 4
        for h in HOSPITALS:
            pend = self._pending(h)
            path = self._path(DEPOT, h)
            o[i] = min(sum(q.units for q in pend), 12) / 12
            o[i + 1] = min(sum(q.units for q in pend if q.urgent), 12) / 12
            o[i + 2] = min((q.deadline - self.t for q in pend), default=24) / 24
            o[i + 3] = 1.0 if path is None else min(self._path_steps(path), 12) / 12
            i += 4
        return np.clip(o, 0, 1)

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sc = SCENARIOS[self.scenario]
        self.t, self._rid = 0, 0
        self.rain = int(self.np_random.choice(4, p=sc["rain_p0"]))
        self.flooded = np.zeros(N_ROADS, dtype=bool)
        self.vans = [Van(v) for v in range(N_VANS)]
        self.requests: list[Request] = []
        self.belief.reset(N_ROADS)
        self.stats = dict(requests=0, on_time=0, late=0, expired=0, units_wasted=0,
                          temp_breaches=0, unsafe_use=0, failed_trips=0, waited_with_urgent=0)
        self.events: list[str] = []
        self.van_seen: set[int] = set()
        self._wasted, self._breach = 0, False
        self._new_requests()
        self._sense()
        return self._obs(), self._info(0, False)

    def _new_requests(self):
        for h in HOSPITALS:
            for _ in range(self.np_random.poisson(REQ_RATE * (1 + 0.3 * self.rain))):
                urg = bool(self.np_random.random() < URGENT_P)
                self.requests.append(Request(self._rid, h, int(self.np_random.integers(1, 5)), urg,
                                             self.t, self.t + DEADLINE_STEPS[urg]))
                self._rid += 1
                self.stats["requests"] += 1

    def _dispatch(self, v, h):
        van = self.vans[v]
        path = self._path(DEPOT, h)
        load, room = [], VAN_CAPACITY
        for q in sorted(self._pending(h), key=lambda q: (not q.urgent, q.deadline)):
            if q.units <= room:
                load.append(q)
                room -= q.units
        for q in load:
            self.requests.remove(q)
        van.busy, van.cargo, van.target, van.returning = True, load, h, False
        van.temp, van.minutes_out = TEMP_START, 0
        van.rise = float(self.np_random.uniform(*TEMP_RISE))
        van.path = path[1:]
        van.node = DEPOT
        van.edge_left = 0
        self.events.append(f"van{v} dispatched to {NODES[h]} with {sum(q.units for q in load)} units")

    def _move_van(self, v):
        """Advance one van by one step. Returns reward from this step."""
        van, r = self.vans[v], 0.0
        if not van.busy:
            return 0.0
        if van.cargo:
            van.temp += van.rise
            van.minutes_out += STEP_MIN
        if van.edge_left == 0 and van.path:
            nxt = van.path[0]
            rid = self.graph[van.node][nxt]["rid"]
            if self.flooded[rid]:                       # tool/route failure: road actually flooded
                self.p_pass[rid] = 0.0                  # the van has now SEEN the flood
                self.van_seen.add(rid)                  # -> reliable report at next sensing
                self.events.append(f"van{v} blocked on R{rid} ({NODES[van.node]}-{NODES[nxt]})")
                new = self._path(van.node, van.target if not van.returning else DEPOT)
                if new is None:                         # no believed route left -> go home
                    if van.cargo:
                        self.stats["failed_trips"] += 1
                        r += R["failed_trip"]
                        self.requests.extend(van.cargo)   # requests go back to the queue
                        van.cargo = []
                        self.events.append(f"van{v} trip FAILED, returning to bank")
                    van.returning = True
                    back = self._path(van.node, DEPOT)
                    van.path = back[1:] if back else []
                    if not back:                        # stranded: simplification - van is recovered to the bank
                        van.busy, van.node, van.path = False, DEPOT, []
                    return r
                van.path = new[1:]
                nxt = van.path[0]
            # heavy rain slows traffic: +25% travel time per rain level
            van.edge_left = int(np.ceil(self.graph[van.node][nxt]["steps"] * (1 + 0.25 * self.rain)))
        if van.edge_left > 0:
            van.edge_left -= 1
            if van.edge_left == 0:
                van.node = van.path.pop(0)
        if not van.path and van.edge_left == 0:
            if van.node == van.target and not van.returning:
                r += self._deliver(v)
                van.returning = True
                back = self._path(van.node, DEPOT) or nx.shortest_path(self.graph, van.node, DEPOT, weight="steps")
                van.path = back[1:]
            elif van.node == DEPOT:
                van.busy, van.returning = False, False
        return r

    def _deliver(self, v):
        van, r = self.vans[v], 0.0
        logger_temp = float(self.readings["box_temp"][v])
        decision = self.safety.decide(logger_temp, van.minutes_out)  # Fuzzy Safety Supervisor
        units = sum(q.units for q in van.cargo)
        if van.temp > TEMP_LIMIT:                       # TRUE breach happened
            self.stats["temp_breaches"] += 1
            self._breach = True
            r += R["temp_breach"]
        if decision in ("quarantine", "discard"):
            self.stats["units_wasted"] += units
            self._wasted += units
            r += R["unit_wasted"] * units
            self.requests.extend(Request(self._rid + i, q.hospital, q.units, q.urgent, q.created, q.deadline)
                                 for i, q in enumerate(van.cargo))   # must be re-sent
            self._rid += len(van.cargo)
            self.events.append(f"van{v} at {NODES[van.target]}: {units} units {decision.upper()} "
                               f"(logger {logger_temp:.1f} C)")
        else:
            if van.temp > TEMP_LIMIT:                   # supervisor said "use" but blood was too warm
                self.stats["unsafe_use"] += 1
                r += R["unsafe_use"]
            for q in van.cargo:
                if self.t <= q.deadline:
                    self.stats["on_time"] += 1
                    r += R["on_time_urgent"] if q.urgent else R["on_time_routine"]
                else:
                    self.stats["late"] += 1
                    r += R["late_urgent"] if q.urgent else R["late_routine"]
            self.events.append(f"van{v} delivered {units} units to {NODES[van.target]}")
        van.cargo = []
        return r

    def step(self, action):
        action = int(action)
        self._wasted, self._breach, self.events = 0, False, []
        reward = 0.0
        # count "lazy" waits: urgent blood is pending and a dispatch was possible, but the agent waited
        if action == 0 and self.action_masks()[1:].any() and any(q.urgent for q in self.requests):
            self.stats["waited_with_urgent"] += 1
        if action > 0:
            v, k = divmod(action - 1, len(HOSPITALS))
            if self.action_masks()[action]:
                self._dispatch(v, HOSPITALS[k])
            else:
                reward -= 1.0                           # should never happen with masking
        # --- world dynamics (one 15-minute tick) ---
        self.t += 1
        sc = SCENARIOS[self.scenario]
        if self.t % 4 == 0:                             # rainfall changes hourly
            u = self.np_random.random()
            if u < sc["up"]:
                self.rain = min(3, self.rain + 1)
            elif u < sc["up"] + sc["down"]:
                self.rain = max(0, self.rain - 1)
        sus = np.array([r_[3] for r_ in ROADS])
        flood_now = self.np_random.random(N_ROADS) < FLOOD_P[self.rain] * sus
        recede = self.np_random.random(N_ROADS) < RECEDE_P[self.rain]
        self.flooded = (self.flooded & ~recede) | flood_now
        for v in range(N_VANS):
            reward += self._move_van(v)
        # expire old requests
        for q in list(self.requests):
            if self.t > q.deadline + EXPIRE_AFTER:
                self.requests.remove(q)
                self.stats["expired"] += 1
                reward += R["expired_urgent"] if q.urgent else R["expired_routine"]
        reward += R["waiting_urgent_per_step"] * sum(q.urgent for q in self.requests)
        self._new_requests()
        self._sense()
        terminated = False
        truncated = self.t >= EPISODE_STEPS
        return self._obs(), float(reward), terminated, truncated, self._info(action, True)

    def _info(self, action, stepped):
        return {
            # --- CONTRACT 6.3 log fields produced by the environment ---
            "belief_var": float(np.mean(self.p_var)),
            "pending_urgent": int(sum(q.urgent for q in self.requests)),
            "units_wasted": int(self._wasted) if stepped else 0,
            "temp_breach": bool(self._breach) if stepped else False,
            "sensor_fault": bool(self.sensor_fault),
            # --- extra (not logged by default; for evaluation / B's memory) ---
            "events": list(self.events),
            "true_flooded": self.flooded.copy(),
            "stats": dict(self.stats),
            "t": self.t,
        }

    def render(self):
        lines = [f"t={self.t * STEP_MIN // 60:02d}:{self.t * STEP_MIN % 60:02d}  rain={self.rain}  "
                 f"flooded={[i for i in range(N_ROADS) if self.flooded[i]]}  pending={len(self.requests)}"]
        for van in self.vans:
            lines.append(f"  van{van.vid}: {'busy' if van.busy else 'idle'} at {NODES[van.node]} "
                         f"cargo={sum(q.units for q in van.cargo)}u temp={van.temp:.1f}C")
        return "\n".join(lines)


gym.register(id="BloodChain-v0", entry_point=BloodChainEnv)
