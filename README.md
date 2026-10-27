# CW2 Group: Agentic Cold-Chain Dispatcher (Kelantan floods)

5023CMD Artificial Intelligence, group project. Read `CW2_Project_Brief.docx` first, then `CONTRACT.md`.

| Folder | Who edits | What |
|---|---|---|
| `C_ml_engine/` | C only | Environment, MaskablePPO training, baselines, evaluation, logger |
| `A_uncertainty/` | A only | Sensor-noise wrapper, Belief Estimator, Fuzzy Safety Supervisor, robustness tests |
| `B_memory_xai/` | B only | Log reader, Memory Gate, Incident VectorDB, Plan Card Generator, Fallback Controller |
| `logs/` | All | Small log excerpts only (`logs/local/` is git-ignored for big logs) |
| `diagrams/` | All | Fig. 1 and Fig. 2 sources and exports |

## 1. Setup (once per laptop, about 10 minutes)
Use **Python 3.11 or 3.12** (3.10–3.13 work; 3.14 does not have all packages yet).

**Windows (PowerShell, inside the repo folder):**
```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_setup.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**macOS / Linux:**
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
```
`check_setup.py` must end with **ALL GOOD**. In VS Code, choose the `.venv` interpreter (Ctrl+Shift+P, then "Python: Select Interpreter").

## 2. Daily Git routine (brief Section 9)
1. Before working: Source Control (Ctrl+Shift+G), then **… → Pull**
2. Edit **only your own folder**
3. After working: write a message like `A: add Beta update to belief estimator`, then **Commit**, then **Sync Changes**

Never commit `.venv/`, models (`*.zip`), TensorBoard runs or long logs. `.gitignore` already blocks them. Share those through Google Drive.

## 3. Quick start (everyone can run these)
```
cd C_ml_engine
python evaluate.py --policy urgent --scenario moderate --episodes 20    # Most-Urgent-First baseline
python evaluate.py --policy nearest --scenario severe --episodes 20     # Nearest-First on unseen severe floods
python train.py --seed 0                                                # MaskablePPO, about 5-10 min on a laptop CPU
tensorboard --logdir runs/tb                                            # open http://localhost:6006
python evaluate.py --policy runs/mppo_lr3e-4_s0/best_model.zip --scenario severe --log
```

## 4. How A and B start NOW (no need to wait for C)
- **A:** see `A_uncertainty/README.md`. Plug your code into the environment through the hooks in `CONTRACT.md`. Test with the `urgent` / `nearest` baselines until C's trained model is ready.
- **B:** see `B_memory_xai/README.md`. Real contract-format logs are already in `logs/` (`sample_urgent_moderate.jsonl`, `sample_urgent_severe.jsonl`), so there is no need to fake them. Switch to the trained-model logs when C shares them (by 28 Oct).

## 5. Status
| Date | Status |
|---|---|
| 21 Oct | v0 environment, logger, baselines and training script pushed. Contract proposed (confirm by 23 Oct) |
| 26 Oct | Checkpoint: is MaskablePPO learning? (go / switch to backup topic) |
| 28 Oct | First trained model + real logs shared with A and B |

> The environment is **synthetic**. Never present its numbers as real Kelantan data.
