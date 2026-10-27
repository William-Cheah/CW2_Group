"""fallback_controller.py (yayuen)
Fallback Controller: wraps a policy and hands control to simpler, more conservative
controllers when the agent's picture of the world becomes unreliable (CONTRACT: B owns this).

    L0  PPO Dispatch Policy -> default
    L1  Most-Urgent-First rule -> uncertainty high (> TAU_ENTER) for ENTER_STEPS in a row
    L2  Blood bank officer (simulated) -> L1 has not settled things within ESCALATE_STEPS while an
                                        urgent request is waiting
                                        -> human takes over (override=True)
    L3  Hold & preserve -> requests are pending and a van is free, but NO dispatch is
                            allowed (every route is believed < 0.5, so C's mask only allows "wait")

Escalation is immediate to the most severe level that applies.
Recovery is stepwise (L3 -> L2 -> L1 -> L0) and needs calm for several steps (hysteresis).
One step = 15 minutes (CONTRACT 6.2)."""

import numpy as np

import log_reader as lr # also puts C_ml_engine on the import path

# ---- Settings (steps of 15 min) ----
ENTER_STEPS = 2        # L0 -> L1 after 30 min of high uncertainty
ESCALATE_STEPS = 4     # L1 -> L2 after 1 h in L1 with an urgent request still waiting
EXIT_STEPS = {1: 4,    # L1 -> L0 after 1 h of low uncertainty
              2: 2,    # L2 -> L1 after 30 min of low uncertainty (officer hands back)
              3: 2}    # L3 -> L2 after 30 min in which a dispatch is possible again
OFFICER_DELAY = 1   # simulated officer needs 1 step to respond after taking over


def uncertainty_from_obs(obs):
    """Mean p(1-p) over the 7 road beliefs. Uses only the observation, so it works with
    any Belief Estimator (equals logged belief_var under C's NaiveBelief)."""
    p = np.asarray(obs[0:lr.N_ROADS], dtype=float)
    return float(np.mean(p * (1 - p)))


class FallbackController:
    def __init__(self, ppo_fn, l1_fn, officer_fn, tau_enter=lr.TAU_ENTER, tau_exit=lr.TAU_EXIT):
        """Each *_fn has C's policy signature: fn(env, obs, rng) -> action."""
        self.ppo_fn, self.l1_fn, self.officer_fn = ppo_fn, l1_fn, officer_fn
        self.tau_enter, self.tau_exit = tau_enter, tau_exit
        self.level_steps = {0: 0, 1: 0, 2: 0, 3: 0} # evidence: time spent at each level
        self.critical_alerts = 0    # urgent request waiting while in L3
        self._env = None

    def _reset(self):
        self.level = 0
        self.high_run = self.low_run = 0    # consecutive steps above tau_enter / below tau_exit
        self.in_level = 0   # steps spent in the current level
        self.unblocked_run = 0  # consecutive steps where the network is NOT blocked
        self.alert_open = False # one critical alert per blockage, not per step

    def _set_level(self, new):
        if new != self.level:
            self.level, self.in_level = new, 0

    def choose(self, env, obs, rng=None):
        """CONTRACT interface: returns (action, fallback_level, override)."""
        if env is not self._env:    # run_episodes makes a new env per episode
            self._env = env
            self._reset()

        # 1. Read the signals
        u = uncertainty_from_obs(obs)
        self.high_run = self.high_run + 1 if u > self.tau_enter else 0
        self.low_run = self.low_run + 1 if u < self.tau_exit else 0
        mask = np.asarray(env.action_masks(), dtype=bool)
        dispatch_possible = bool(mask[1:].any())
        pending = bool(np.any(np.asarray(obs[18:34:4]) > 0))
        urgent = bool(np.any(np.asarray(obs[19:34:4]) > 0))
        van_free = bool(obs[10] > 0.5 or obs[14] > 0.5) # idle flag of van 0 / van 1
        # "blocked" = work waiting, a van ready, yet every route is masked as flooded.
        # (If both vans are simply busy, no dispatch is possible either, but that is NOT a hold.)
        blocked = pending and van_free and not dispatch_possible
        self.unblocked_run = self.unblocked_run + 1 if not blocked else 0

        # 2. Which level do the conditions call for? (checked most severe first)
        if blocked:
            target = 3
        elif self.level >= 1 and self.in_level >= ESCALATE_STEPS and urgent:
            target = 2
        elif self.high_run >= ENTER_STEPS:
            target = 1
        else:
            target = 0

        # 3. Escalate at once; recover one level at a time after enough calm (hysteresis)
        if target > self.level:
            self._set_level(target)
        elif target < self.level:
            calm = {1: self.low_run, 2: self.low_run, 3: self.unblocked_run}[self.level]
            # calm must also have lasted inside THIS level, so each level is held for a while
            if min(calm, self.in_level) >= EXIT_STEPS[self.level]:
                self._set_level(self.level - 1)
        self.in_level += 1
        self.level_steps[self.level] += 1

        # 4. Act according to the level
        override = False
        if self.level == 0:
            action = self.ppo_fn(env, obs, rng)
        elif self.level == 1:
            action = self.l1_fn(env, obs, rng)
        elif self.level == 2:
            override = True
            action = 0 if self.in_level <= OFFICER_DELAY else self.officer_fn(env, obs, rng)
        else:
            action = 0  # hold: keep stock in cold storage
            if urgent and not self.alert_open:  # life-critical -> senior supervisor
                self.critical_alerts += 1
                self.alert_open = True
        if self.level != 3:
            self.alert_open = False

        if not mask[int(action)]:   # never send an invalid action
            action = 0
        return int(action), self.level, override

    def __call__(self, env, obs, rng):
        """Lets C's run_episodes use the controller like any other policy."""
        return self.choose(env, obs, rng)[0]

    def level_share(self):
        total = max(1, sum(self.level_steps.values()))
        return {f"L{k}": round(v / total, 3) for k, v in self.level_steps.items()}


if __name__ == "__main__":
    # Evidence: the same policy WITH and WITHOUT the Fallback Controller, on unseen severe floods.
    import argparse
    from evaluate import make_policy, run_episodes, summarise, CFG

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="nearest", help="L0 policy: nearest | urgent | path/to/best_model.zip")
    ap.add_argument("--scenario", default="severe", choices=["moderate", "severe"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=None, help="first seed; 5000 = validation (for tuning)")
    ap.add_argument("--tau-enter", type=float, default=lr.TAU_ENTER)
    ap.add_argument("--tau-exit", type=float, default=lr.TAU_EXIT)
    ap.add_argument("--enter-steps", type=int, default=ENTER_STEPS)
    ap.add_argument("--no-l2", action="store_true", help="disable escalation to the officer")
    args = ap.parse_args()

    label, l0 = make_policy(args.policy)
    _, l1 = make_policy("urgent")
    _, officer = make_policy("nearest")
    ENTER_STEPS = args.enter_steps
    if args.no_l2:
        ESCALATE_STEPS = 10**9  # never escalates
    seed0 = args.seed0 if args.seed0 is not None else CFG["seeds"]["eval_start"]

    base = run_episodes(l0, args.scenario, args.episodes, seed0)
    ctrl = FallbackController(l0, l1, officer, args.tau_enter, args.tau_exit)
    wrapped = run_episodes(ctrl, args.scenario, args.episodes, seed0)

    print(summarise(base, label, args.scenario))
    print(summarise(wrapped, label + "+fallback", args.scenario))
    print("Time at each level:", ctrl.level_share(), "| critical alerts:", ctrl.critical_alerts)
    diff = wrapped["ret"].values - base["ret"].values      # same seeds -> compare day by day
    ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"Paired return difference (fallback - base): {diff.mean():+.1f} ± {ci:.1f} (95% CI)")
    better = (diff > 0).mean()
    print(f"Fallback better on {better:.0%} of days")