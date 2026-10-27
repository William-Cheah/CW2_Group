"""incident_db.py (yayuen)
Incident VectorDB: stores past incidents as 14-number vectors (from log_reader.to_vector)
and finds the most similar ones with FAISS.

Design choice: Euclidean (L2) distance, NOT cosine similarity. On our data, cosine rated
"road 0 probably open" vs "road 0 flooded" as 0.955 similar, so it would merge situations
that need different decisions. L2 distance separates them (0.79).
"""
from pathlib import Path
import json

import faiss
import numpy as np

# --- Settings (calibrate from data) ---
DIM = 14    # length of log_reader.to_vector
DEFAULT_K = 3   # how many precedents to return
MAX_DISTANCE = 0.5  # further than this = "not similar", not returned

def _as_row(vector):
    """FAISS wants a 2-D float32 array: 1 row x DIM columns."""
    v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if v.shape[1] != DIM:
        raise ValueError(f"vector has {v.shape[1]} values, expected {DIM}")
    return v


class IncidentDB:
    def __init__(self, dim=DIM):
        self.index = faiss.IndexFlatL2(dim) # exact search; fine for hundreds/thousands of incidents
        self.records = []   # metadata, same order as vectors in the index

    def __len__(self):
        return len(self.records)

    def add(self, vector, meta):
        """Store one incident. meta = dict with action, outcome, etc. Returns its id."""
        self.index.add(_as_row(vector))
        record = dict(meta)
        record["id"] = len(self.records)
        record.setdefault("count", 1)   # how many similar incidents were merged into this one
        record.setdefault("contested", False)   # True if similar incidents had opposite outcomes
        self.records.append(record)
        return record["id"]

    def search(self, vector, k=DEFAULT_K, max_distance=MAX_DISTANCE):
        """Return up to k (distance, record) pairs, closest first, within max_distance."""
        if len(self) == 0:
            return []
        sq_dist, ids = self.index.search(_as_row(vector), min(k, len(self)))
        results = []
        for d2, i in zip(sq_dist[0], ids[0]):
            d = float(np.sqrt(max(d2, 0.0)))   # IndexFlatL2 returns SQUARED distances
            if i != -1 and d <= max_distance:
                results.append((d, self.records[i]))
        return results

    def update(self, record_id, **changes):
        """Change metadata of a stored incident (used by the Memory Gate to merge / flag)."""
        self.records[record_id].update(changes)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "incidents.faiss"))
        (folder / "incidents.json").write_text(json.dumps(self.records, indent=2, default=float))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        db = cls()
        db.index = faiss.read_index(str(folder / "incidents.faiss"))
        db.records = json.loads((folder / "incidents.json").read_text())
        return db


if __name__ == "__main__":
    # Self-test on real log data
    import log_reader as lr

    df = lr.load(lr.ROOT / "logs" / "sample_urgent_severe.jsonl")
    vec = lambda i: lr.to_vector(df.loc[i, "obs"], df.loc[i, "belief_var"])

    db = IncidentDB()
    db.add(vec(0), {"note": "step 1"})
    db.add(vec(1), {"note": "step 2 (road 0 flooded)"})
    db.add(vec(2), {"note": "step 3 (road 5 flooded, urgent request)"})
    print(f"Stored {len(db)} incidents")

    print("\nSearch with step 1 itself (expect step 1 at distance 0):")
    for d, r in db.search(vec(0), max_distance=10):
        print(f"  distance {d:.3f}  ->  {r['note']}")

    print(f"\nSame search, only matches within MAX_DISTANCE={MAX_DISTANCE}:")
    for d, r in db.search(vec(0)):
        print(f"  distance {d:.3f}  ->  {r['note']}")

    db.save(lr.ROOT / "logs" / "local" / "incident_db_test")
    db2 = IncidentDB.load(lr.ROOT / "logs" / "local" / "incident_db_test")
    print(f"\nSaved and reloaded: {len(db2)} incidents")
    