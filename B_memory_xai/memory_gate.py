"""memory_gate.py (yayuen)
Memory Gate: decides which incidents enter long-term memory (the Incident VectorDB).

Rules, applied in order:
    1. RESOLVED     the full outcome window was observed
                    else -> rejected_unresolved
    2. SIGNIFICANT  |outcome_reward| >= MIN_IMPACT
                    else -> rejected_insignificant
    3. CONFIDENT    belief_var at the end <= TAU_EXIT, or a human
                    else -> still stored, but flagged provisional 
                    (lower weight on retrieval)
    4. NOVEL        no stored incident within DUP_DISTANCE
                    else -> merged (same outcome) or contested (opposite outcome)
Passing rules 1, 2 and 4 -> stored (flagged provisional if rule 3 fails)."""

from collections import Counter

import pandas as pd

import log_reader as lr
from incident_db import IncidentDB

# ---- Settings ----
MIN_IMPACT = 3.0    # |outcome_reward| below this = nothing worth learning (about one late urgent delivery)
DUP_DISTANCE = 0.2  # closer than this (L2) = the same situation
HARM_REWARD = -3.0  # outcome_reward at or below this = harmful


def outcome_label(inc):
    """'harmful' or 'good': the lesson this incident teaches."""
    harmful = (inc["outcome_reward"] <= HARM_REWARD
               or inc["outcome_wasted"] > 0
               or bool(inc["outcome_breach"]))
    return "harmful" if harmful else "good"


class MemoryGate:
    def __init__(self, db=None):
        self.db = db if db is not None else IncidentDB()
        self.counts = Counter()

    def consider(self, inc):
        """Apply the four rules to one incident (a row from log_reader.find_incidents)."""
        if not inc["resolved"]:
            return self._count("rejected_unresolved")
        if abs(inc["outcome_reward"]) < MIN_IMPACT:
            return self._count("rejected_insignificant")
        # Uncertain incidents are NOT rejected: the worst floods are also the most uncertain,
        # so rejecting them would leave memory knowing least about the most dangerous cases.
        provisional = bool(inc["end_belief_var"] > lr.TAU_EXIT and not inc["override"])
        tag = "_provisional" if provisional else ""

        vector = lr.to_vector(inc["start_obs"], inc["start_belief_var"])
        label = outcome_label(inc)
        nearest = self.db.search(vector, k=1, max_distance=DUP_DISTANCE)

        if nearest: # same situation already in memory
            _, old = nearest[0]
            if old["label"] == label:   # same lesson -> merge, keep store compact
                n = old["count"] + 1
                avg = old["outcome_reward"] + (inc["outcome_reward"] - old["outcome_reward"]) / n
                self.db.update(old["id"], count=n, outcome_reward=avg,
                               provisional=old.get("provisional", False) and provisional)
                return self._count("merged" + tag)
            # opposite lesson -> keep both, flag both, never silently overwrite evidence
            self.db.update(old["id"], contested=True)
            self.db.add(vector, self._meta(inc, label, provisional, contested=True))
            return self._count("contested" + tag)

        self.db.add(vector, self._meta(inc, label, provisional))
        return self._count("stored" + tag)

    def _meta(self, inc, label, provisional, contested=False):
        return {
            "episode": int(inc["episode"]), "start_step": int(inc["start_step"]),
            "action": inc["start_action"], "label": label,
            "outcome_reward": float(inc["outcome_reward"]),
            "outcome_wasted": int(inc["outcome_wasted"]),
            "outcome_breach": bool(inc["outcome_breach"]),
            "contested": contested,
            "provisional": provisional,
        }

    def _count(self, decision):
        self.counts[decision] += 1
        return decision


def run_gate(log_paths, db=None):
    """Feed every incident from one or more logs through the gate. Returns (gate, decisions table)."""
    gate = MemoryGate(db)
    rows = []
    for path in log_paths:
        incidents = lr.find_incidents(lr.load(path))
        for _, inc in incidents.iterrows():
            decision = gate.consider(inc)
            rows.append({"log": path.name, "episode": inc["episode"], "start_step": inc["start_step"],
                         "outcome_reward": round(inc["outcome_reward"], 1), "decision": decision})
    return gate, pd.DataFrame(rows)


if __name__ == "__main__":
    logs = [lr.ROOT / "logs" / "sample_urgent_moderate.jsonl",
            lr.ROOT / "logs" / "sample_urgent_severe.jsonl"]
    logs = [p for p in logs if p.exists()]
    gate, table = run_gate(logs)
    print(table.to_string(index=False))
    print("\nMemory Gate decisions:")
    for decision, n in gate.counts.most_common():
        print(f"  {decision:<24}{n}")
    print(f"\nIncidents in long-term memory: {len(gate.db)}")
    gate.db.save(lr.ROOT / "logs" / "local" / "incident_db")