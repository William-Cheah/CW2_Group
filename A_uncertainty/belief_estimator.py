"""Belief Estimator (Member A) - plugs into env.unwrapped.belief (CONTRACT 6.2).

Bayesian estimate of whether each road is passable, using a Beta-Bernoulli model:
    alpha[r] = evidence that road r is OPEN
    beta[r]  = evidence that road r is FLOODED
    p_passable[r] = alpha / (alpha + beta)                 (mean of Beta)
    var[r]        = alpha*beta / ((a+b)^2 * (a+b+1))        (variance of Beta)

Every step:
  1. decay  - old evidence fades toward the prior (roads flood and reopen)
  2. reports - field reports (90% correct) add evidence
  3. van_seen_flooded - a van hit the road: strong, reliable evidence of flooding
  4. rain   - heavy rain is weak evidence of flooding on every road
"""
import numpy as np


class BetaBelief:
    def __init__(self, prior_alpha=1.0, prior_beta=1.0, decay=0.9,
                 report_weight=1.0, van_weight=5.0, rain_weight=0.1):
        self.prior_a = prior_alpha      # Beta(1,1) = uniform = "no knowledge"
        self.prior_b = prior_beta
        self.decay = decay              # per 15-min step; 0.9 -> evidence half-life ~6.6 steps (~1.6 h)
        self.report_weight = report_weight  # field report (90% correct)
        self.van_weight = van_weight        # van physically hit a flooded road (reliable)
        self.rain_weight = rain_weight      # weak evidence per rain level (0-3)
        self.alpha = None
        self.beta = None

    # ---- CONTRACT hook 1 ----
    def reset(self, n_roads):
        self.alpha = np.full(n_roads, self.prior_a, dtype=float)
        self.beta = np.full(n_roads, self.prior_b, dtype=float)

    # ---- CONTRACT hook 2 ----
    def update(self, readings):
        # 1. decay toward the prior
        self.alpha = self.prior_a + (self.alpha - self.prior_a) * self.decay
        self.beta = self.prior_b + (self.beta - self.prior_b) * self.decay

        # 2. field reports: 1 = open, 0 = flooded, -1 = no report
        reports = np.asarray(readings["road_reports"])
        self.alpha[reports == 1] += self.report_weight
        self.beta[reports == 0] += self.report_weight

        # 3. a van hit a flooded road -> strong evidence
        for road in readings.get("van_seen_flooded", []):
            self.beta[road] += self.van_weight

        # 4. rain level 0-3 -> weak flood evidence on all roads
        self.beta += self.rain_weight * readings["rain_level"]

        return self.p_passable(), self.var()

    def p_passable(self):
        return self.alpha / (self.alpha + self.beta)

    def var(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


if __name__ == "__main__":
    # Stand-alone test with fake readings (no environment needed)
    be = BetaBelief()
    be.reset(n_roads=7)
    for t in range(8):
        reports = np.full(7, -1)
        reports[0] = 1                       # road 0: always reported open
        reports[1] = 0                       # road 1: always reported flooded
        # road 2: never reported
        readings = {"rain_level": 2, "river_level": 0.5, "road_reports": reports,
                    "box_temp": [4.0, 4.0], "van_seen_flooded": [3] if t == 4 else []}
        p, v = be.update(readings)
    for r in range(4):
        print(f"road {r}: alpha={be.alpha[r]:.2f} beta={be.beta[r]:.2f} "
              f"p_passable={p[r]:.2f} var={v[r]:.4f}")
    print("mean var (this is belief_var in the log):", round(float(v.mean()), 4))