"""Fuzzy Safety Supervisor (Member A) - plugs into env.unwrapped.safety (CONTRACT 6.2).

Contract signature:  decide(logger_temp_c, minutes_out) -> "use" | "quarantine" | "discard"

Mamdani fuzzy inference with scikit-fuzzy (Zadeh, 1965; Mamdani and Assilian, 1975):
  1. fuzzify   - how "in range / near limit / over limit / too cold" is the logger reading,
                 how "short / long" was the time out of controlled storage
  2. rules     - IF temperature is A AND time is B THEN decision is C   (AND = min)
  3. aggregate - clip each decision set at its rule strength, combine with max
  4. defuzzify - centroid -> a risk score 0..100 -> use (< 40) / quarantine (< 65) / discard

Why the "near limit" set starts 1.2 C below the +10 C transport limit (CONTRACT 6.2):
  the decision uses the LOGGER reading, which (a) has noise sd 0.3 C -> 2 sd = 0.6 C, and
  (b) was taken one 15-minute step BEFORE delivery, while the box warms 0.2-0.6 C per step.
  0.6 + 0.6 = 1.2 C, so a box truly at 10 C can read as low as 8.8 C.
Thresholds are design values for our synthetic environment, not clinical guidance.
"""
import numpy as np
import skfuzzy as fuzz

TEMP_LIMIT = 10.0                       # transport limit (CONTRACT 6.2; cite WHO 2005 in the report)
T = np.arange(0.0, 16.01, 0.05)         # logger temperature universe, deg C
M = np.arange(0.0, 480.1, 1.0)          # minutes out of controlled storage
R = np.arange(0.0, 100.1, 0.5)          # risk score universe

# ---- membership functions (design values, see docstring) ----
TEMP = {
    "too_cold":   fuzz.trapmf(T, [0.0, 0.0, 1.5, 2.0]),     # below +2 C: freezing risk
    "in_range":   fuzz.trapmf(T, [1.5, 2.0, 8.4, 8.8]),     # safe band
    "near_limit": fuzz.trapmf(T, [8.4, 8.8, 9.8, 10.4]),    # within 1.2 C of the limit
    "over_limit": fuzz.trapmf(T, [9.8, 10.4, 16.0, 16.0]),  # beyond the limit
}
TIME = {
    "short": fuzz.trapmf(M, [0, 0, 180, 300]),              # up to 3-5 h: within a cold box's hold time
    "long":  fuzz.trapmf(M, [240, 360, 480, 480]),          # 4-6 h and more (C's CrispSafety also uses 4 h)
}
DECISION = {
    "use":        fuzz.trimf(R, [0, 0, 40]),
    "quarantine": fuzz.trimf(R, [30, 50, 70]),
    "discard":    fuzz.trimf(R, [60, 100, 100]),
}
# ---- rule base (Table 2 in the report) ----
RULES = [
    ("R1", "in_range",   "short", "use"),
    ("R2", "in_range",   "long",  "quarantine"),
    ("R3", "near_limit", "short", "quarantine"),
    ("R4", "near_limit", "long",  "discard"),
    ("R5", "over_limit", None,    "discard"),     # None = any time
    ("R6", "too_cold",   None,    "quarantine"),
]
USE_BELOW, QUARANTINE_BELOW = 40.0, 65.0


class FuzzySafety:
    def __init__(self):
        self.last = None                 # explanation of the last decision (for plan cards / Viva)

    def score(self, logger_temp_c, minutes_out):
        t = float(np.clip(logger_temp_c, T[0], T[-1]))
        m = float(np.clip(minutes_out, M[0], M[-1]))
        mu_t = {k: fuzz.interp_membership(T, mf, t) for k, mf in TEMP.items()}
        mu_m = {k: fuzz.interp_membership(M, mf, m) for k, mf in TIME.items()}
        agg = np.zeros_like(R)
        fired = {}
        for name, tk, mk, out in RULES:
            strength = mu_t[tk] if mk is None else min(mu_t[tk], mu_m[mk])     # AND = min
            fired[name] = round(float(strength), 3)
            agg = np.fmax(agg, np.fmin(strength, DECISION[out]))             # clip + max
        if agg.max() == 0:              # cannot happen inside the universes; fail safe
            return 100.0, fired
        return float(fuzz.defuzz(R, agg, "centroid")), fired

    # ---- CONTRACT hook ----
    def decide(self, logger_temp_c, minutes_out):
        s, fired = self.score(logger_temp_c, minutes_out)
        d = "use" if s < USE_BELOW else ("quarantine" if s < QUARANTINE_BELOW else "discard")
        self.last = dict(logger_temp_c=logger_temp_c, minutes_out=minutes_out,
                         score=round(s, 1), decision=d, rules=fired)
        return d


if __name__ == "__main__":
    fs = FuzzySafety()
    print(f"{'logger C':>8} {'min out':>7} {'score':>6}  decision    strongest rule")
    for temp, mins in [(5.0, 30), (5.0, 200), (8.0, 60), (8.6, 60), (9.0, 60), (9.0, 200),
                       (9.6, 30), (9.9, 30), (10.5, 30), (1.5, 30)]:
        d = fs.decide(temp, mins)
        top = max(fs.last["rules"], key=fs.last["rules"].get)
        print(f"{temp:8.1f} {mins:7d} {fs.last['score']:6.1f}  {d:10s}  {top} ({fs.last['rules'][top]})")
