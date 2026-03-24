# SPEC.md — Agentic ML Learning Loop: Technical Specification

This document resolves every open question from `PLAN.md` and defines the exact file contents, interfaces, and behavior required for implementation.

---

## 1. File Inventory

| File | Status | Description |
|---|---|---|
| `data/dataset.csv` | Exists | 100-row tabular dataset |
| `evaluate.py` | To create | Fixed evaluation harness — agent cannot edit |
| `train.py` | To create | Baseline model — agent edits only this file |
| `program.md` | To create | Natural-language instructions for the agent's loop |
| `results.tsv` | Auto-generated | Structured experiment log (TSV), gitignored |
| `experiment_log.md` | Auto-generated | Detailed narrative journal of every experiment, gitignored |
| `run.log` | Auto-generated | Raw stdout/stderr of the latest experiment run, gitignored |

---

## 2. Dataset

### 2.1 File

`data/dataset.csv` — 100 rows, committed, never modified.

Columns: `x1`, `x2`, `y` (all floats, rounded to 4 decimal places).

### 2.2 Generating Function

The true function (not revealed to the agent — it only sees the CSV):

```
y = 3·x1² + 2·sin(x2) + 1.5·x1·x2 + ε
```

Where:
- `x1`, `x2` ~ Uniform(-3, 3), `seed=42`
- `ε` ~ Normal(0, σ²), `σ = 0.5`

Properties:
- Non-linear: `x1²` dominates the output and has high variance, so a linear model performs very poorly (baseline MSE ≈ 76)
- Interaction term: `x1·x2` rewards feature engineering
- Periodic component: `sin(x2)` rewards non-polynomial transformations
- Noise floor: theoretical minimum MSE ≈ 0.25 (= σ² = 0.5²)

### 2.3 Train/Validation Split

Fixed in `evaluate.py`, never changed:

```python
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# -> 80 train rows, 20 val rows
```

---

## 3. `evaluate.py` — Evaluation Harness (Fixed)

The trusted measurement contract. The agent must not read or modify this file.

### 3.1 Complete Implementation

```python
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/dataset.csv"
RANDOM_SEED = 42


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def evaluate(predictions: np.ndarray, val_y: np.ndarray) -> float:
    """Compute MSE. Returns float('inf') on any error."""
    try:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != val_y.shape:
            return float("inf")
        if not np.all(np.isfinite(predictions)):
            return float("inf")
        return float(mean_squared_error(val_y, predictions))
    except Exception:
        return float("inf")


def main():
    import train  # agent-editable module

    train_X, val_X, train_y, val_y = load_and_split()

    t0 = time.perf_counter()
    try:
        predictions = train.train_and_predict(train_X, train_y, val_X)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"CRASH: {e}")
        print(f"val_mse:          inf")
        print(f"elapsed_seconds:  {elapsed:.1f}")
        print(f"num_rows_train:   {len(train_X)}")
        print(f"num_rows_val:     {len(val_X)}")
        return

    elapsed = time.perf_counter() - t0
    val_mse = evaluate(predictions, val_y)

    print(f"val_mse:          {val_mse:.6f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"num_rows_train:   {len(train_X)}")
    print(f"num_rows_val:     {len(val_X)}")


if __name__ == "__main__":
    main()
```

### 3.2 Error Handling Policy

| Situation | Behavior |
|---|---|
| `train_and_predict` raises an exception | Print `CRASH: <message>`, print `val_mse: inf` |
| Predictions contain NaN or Inf | `evaluate()` returns `float('inf')` |
| Predictions wrong shape | `evaluate()` returns `float('inf')` |
| Predictions wrong type | Coerced via `np.asarray`; if still broken, returns `float('inf')` |

The evaluator **never crashes**. It always prints the metric line so the agent can always parse a result.

### 3.3 Output Format

Exact stdout format (fixed field width for readability):

```
val_mse:          0.042317
elapsed_seconds:  1.3
num_rows_train:   80
num_rows_val:     20
```

The agent parses `val_mse` by grepping for the line starting with `val_mse:` and reading the float value after the colon.

---

## 4. `train.py` — Model File (Agent-Editable)

### 4.1 Required Interface

`train.py` must expose exactly one function:

```python
def train_and_predict(
    train_X: np.ndarray,  # shape (80, 2)
    train_y: np.ndarray,  # shape (80,)
    val_X:   np.ndarray,  # shape (20, 2)
) -> np.ndarray:          # shape (20,), dtype float
```

Constraints:
- Must return a 1D numpy array of length 20
- Must not read files, make network requests, or spawn subprocesses
- Must not import from `evaluate.py`
- Should set internal random seeds for reproducibility

### 4.2 Baseline Implementation

The starting `train.py` for the first commit:

```python
import numpy as np
from sklearn.linear_model import LinearRegression


def train_and_predict(train_X, train_y, val_X):
    model = LinearRegression()
    model.fit(train_X, train_y)
    return model.predict(val_X)
```

Expected baseline `val_mse`: approximately **70–80** (linear regression cannot capture `x1²` which dominates the output and has high variance).

### 4.3 Allowed Dependencies

The agent may import any of the following in `train.py`:

- `numpy`
- `scipy`
- `scikit-learn` (all of it: `sklearn.*`)
- `xgboost`
- `lightgbm`
- Standard library: `math`, `itertools`, `functools`, `random`

Forbidden:
- `requests`, `urllib`, `httpx`, or any HTTP library
- `subprocess`, `os.system`, or process-spawning
- `open()` / file I/O of any kind
- `evaluate`, `data`, or any project-local module other than numpy/scipy/sklearn/xgboost/lightgbm

---

## 5. `program.md` — Agent Instructions

This file governs the agent's autonomous behavior. Full content:

```markdown
# program.md — Agentic ML Loop Instructions

You are an autonomous ML optimization agent. Your goal: minimize `val_mse` on
the validation set by iteratively improving `train.py`.

You maintain two log files that together provide a complete record of your work:
- `results.tsv` — structured data: every experiment's number, commit, metric,
  comparison to previous best, timing, status, model type, and description.
- `experiment_log.md` — narrative journal: your reasoning, hypothesis, what you
  changed, what happened, what you learned, and what you plan to try next.

These logs are the primary deliverable of your work — a human should be able to
read `experiment_log.md` and fully understand your decision-making process.

## Setup (run once at the start)

1. Ensure you are on branch `agentic-ml/run1` (create it from main if needed).
2. Initialize `results.tsv` with the header row:
   ```
   experiment\tcommit\tval_mse\tprev_best_mse\timprovement\telapsed_sec\tstatus\tmodel_type\tdescription
   ```
3. Initialize `experiment_log.md` with the header block:
   ```markdown
   # Experiment Log

   This file records every experiment run by the agentic ML loop, including
   the reasoning behind each attempt, the outcome, and lessons learned.

   **Dataset:** 100 rows, 2 features (x1, x2), 1 target (y)
   **Metric:** val_mse (lower is better)
   **Noise floor:** ~0.25 (theoretical minimum, σ²=0.5²)
   ```
4. Run `python evaluate.py > run.log 2>&1` to establish the baseline.
5. Read `run.log`. Extract `val_mse` and `elapsed_seconds`.
6. Append the baseline row to `results.tsv` and the baseline entry to
   `experiment_log.md` (see logging format below).
7. Print the baseline MSE. Begin the loop.

## The Loop

Repeat forever:

### Step 1: Hypothesize
- Read `experiment_log.md` (especially the "Next direction" from the last entry)
  and `results.tsv` to understand what has been tried and what worked.
- Look at the current `train.py`.
- Form a specific, testable hypothesis. State it clearly — you will write it
  down in the log before running the experiment.
- Avoid repeating hypotheses that have already been tried and discarded.

### Step 2: Implement
- Edit `train.py` only. Do not touch any other file.
- Keep the function signature `train_and_predict(train_X, train_y, val_X)` intact.
- Set random seeds inside your code for reproducibility.

### Step 3: Commit
- `git add train.py`
- `git commit -m "<short description of the hypothesis>"`

### Step 4: Run
- `python evaluate.py > run.log 2>&1`
- Time budget: 30 seconds. If the process has not finished in 60 seconds, kill it.
  Treat this as a CRASH.

### Step 5: Parse Result
- Read `run.log`.
- Extract the float after `val_mse:` and `elapsed_seconds:`.
- If no `val_mse:` line exists → CRASH.

### Step 6: Keep or Discard
- Let `prev_mse` = the best `val_mse` in `results.tsv` with status `keep`.
- Compute `improvement` = percentage change from `prev_mse` to `new_mse`.
- **KEEP** if `new_mse < prev_mse` (strict improvement):
  - Continue to Step 7 with `status = keep`.
- **DISCARD** if `new_mse >= prev_mse`:
  - `git reset --hard HEAD~1`
  - Continue to Step 7 with `status = discard`.
- **CRASH** (no metric or exception):
  - `git reset --hard HEAD~1`
  - Read the tail of `run.log`. Understand the error.
  - Continue to Step 7 with `status = crash`.

### Step 7: Log (MANDATORY — do this after EVERY experiment)

**7a. Append to `results.tsv`:**
One row with all 9 columns:
```
<experiment_num>\t<commit>\t<val_mse>\t<prev_best_mse>\t<improvement>\t<elapsed_sec>\t<status>\t<model_type>\t<description>
```

**7b. Append to `experiment_log.md`:**
A full narrative entry with ALL SIX required sections:

```markdown
---

### Experiment <N> — <status>
**Hypothesis:** <one sentence: what you are testing>

**Reasoning:** <why you believe this will improve the metric, referencing
evidence from prior experiments>

**What changed in `train.py`:** <plain-English description of the code
changes — not a diff, but enough that a reader understands what was modified>

**Result:** val_mse = <value> (prev best: <value>, change: <+/-%>)
Elapsed: <seconds>s

**Conclusion:** <what you learned — did the result confirm or refute the
hypothesis? why? what does this tell you about the data or the model?>

**Next direction:** <what you plan to try next based on what you learned>
```

Do not skip or abbreviate any section. The narrative log is the primary record
of your reasoning. A human reading only `experiment_log.md` should understand
every decision you made and why.

**Return to Step 1.**

## Strategy Guidelines

- **Think before coding.** A hypothesis should predict *why* a change will help.
- **Learn from history.** Before each experiment, read `experiment_log.md`.
  The "Next direction" from your last entry is your starting point. Do not
  repeat discarded approaches unless you have a specific reason to revisit them
  (and you must explain that reason in the log).
- **Explore first, exploit later.** Try diverse approaches early (feature
  engineering, model family changes, hyperparameter tuning). Tune within a
  promising direction once you've identified it.
- **Favor simplicity.** A smaller improvement with simpler code is better than
  a large improvement with brittle code.
- **Value deletions.** If removing code maintains performance, that is an
  improvement.
- **Explain your thinking.** The experiment log should read like a research
  notebook. Connect each experiment to what came before. Build a coherent
  narrative, not a random walk.
- **The noise floor is ~0.25.** `val_mse` cannot go below this. Below ~0.4 is excellent.

## Constraints

- Never edit `evaluate.py` or `data/dataset.csv`.
- Never hardcode validation set answers.
- Never make the function read from files or the network.
- Keep `train.py` self-contained.
- Always write the full log entry for every experiment — no exceptions.
```

---

## 6. `results.tsv` — Structured Experiment Log

### 6.1 Format

Tab-separated, no quoting. Header row on first line:

```
experiment	commit	val_mse	prev_best_mse	improvement	elapsed_sec	status	model_type	description
1	a1b2c3d	5.847123	-	-	0.1	keep	LinearRegression	baseline linear regression
2	b2c3d4e	2.013847	5.847123	-65.6%	0.2	keep	LinearRegression	add degree-2 polynomial features
3	c3d4e5f	2.198400	2.013847	+9.2%	0.3	discard	DecisionTree	switch to decision tree depth=5 (overfit)
4	d4e5f6g	inf	2.013847	-	-	crash	XGBRegressor	import error in xgboost params
5	e5f6g7h	0.312045	2.013847	-84.5%	0.8	keep	GradientBoosting	gradient boosting n_estimators=200 lr=0.05
```

Fields:
- `experiment`: sequential integer starting at 1 (never resets, even across discards)
- `commit`: first 7 characters of the git commit hash
- `val_mse`: float, 6 decimal places (or `inf` for crashes)
- `prev_best_mse`: the best `val_mse` with status `keep` before this experiment (or `-` for the baseline)
- `improvement`: percentage change from `prev_best_mse` (negative = better, positive = worse, `-` for baseline/crash)
- `elapsed_sec`: wall-clock seconds for `train_and_predict` to run (or `-` for crashes)
- `status`: one of `keep`, `discard`, `crash`
- `model_type`: the primary model class used (e.g., `LinearRegression`, `GradientBoosting`, `XGBRegressor`, `Ensemble`)
- `description`: free-form, matches the git commit message

### 6.2 Why These Columns

The `experiment` counter and `prev_best_mse`/`improvement` columns let a human (or the agent itself) quickly scan the history and understand:
- How many experiments have been run total
- What the best MSE was at the time each experiment was tried
- Whether a discarded approach was close to improving or wildly off
- Which model families have been explored
- How runtime scales with model complexity

---

## 7. `experiment_log.md` — Detailed Narrative Journal

This is the most important logging artifact. It provides a **human-readable, chronological narrative** of every experiment — the reasoning, the outcome, and the lessons learned. The agent appends to this file after every experiment.

### 7.1 Purpose

`results.tsv` answers "what happened?" — `experiment_log.md` answers "why did the agent do it, what did it expect, and what did it learn?" This is critical for:
- **Debugging**: understanding why the agent went down a particular path
- **Learning**: seeing what strategies worked and what failed, and why
- **Transparency**: a human can read this file and fully understand the agent's decision-making process without reading any code

### 7.2 Entry Format

Each experiment gets a full entry. The agent must write this **after** the keep/discard decision:

```markdown
---

### Experiment 3 — discard
**Hypothesis:** Switch from polynomial-augmented linear regression to a decision tree (depth=5) to better capture non-linear boundaries.

**Reasoning:** The polynomial features improved MSE from 5.85 to 2.01, confirming non-linearity in the data. A decision tree may find the non-linear structure more naturally without needing hand-crafted features.

**What changed in `train.py`:** Replaced `LinearRegression` with `DecisionTreeRegressor(max_depth=5, random_state=42)`. Removed the polynomial feature preprocessing.

**Result:** val_mse = 2.198400 (prev best: 2.013847, change: +9.2%)

**Conclusion:** Decision tree with depth=5 overfit the training set. The polynomial features in the linear model were actually capturing the structure more robustly. A tree-based model may still work but needs regularization (lower depth, min_samples_leaf) or should be combined with the polynomial features rather than replacing them.

**Next direction:** Try gradient boosting with shallow trees (depth=2–3) which ensembles many weak learners and naturally regularizes.
```

### 7.3 Required Sections Per Entry

Every entry MUST include all six sections:

1. **Hypothesis** — one sentence: what the agent is testing
2. **Reasoning** — why the agent believes this will improve the metric, referencing evidence from prior experiments
3. **What changed in `train.py`** — a plain-English description of the code changes (not a diff, but enough that a reader understands what was modified)
4. **Result** — the `val_mse`, the previous best, and the percentage change
5. **Conclusion** — what the agent learned from the result, whether it confirmed or refuted the hypothesis, and why
6. **Next direction** — what the agent plans to try next based on what it learned (this creates a reasoning chain across experiments)

### 7.4 Baseline Entry

The first entry is special:

```markdown
# Experiment Log

This file records every experiment run by the agentic ML loop, including the reasoning behind each attempt, the outcome, and lessons learned.

**Dataset:** 100 rows, 2 features (x1, x2), 1 target (y)
**Metric:** val_mse (lower is better)
**Noise floor:** ~0.09 (theoretical minimum)

---

### Experiment 1 — keep (baseline)
**Hypothesis:** Establish a baseline using simple linear regression with no feature engineering.

**Reasoning:** A linear model provides the simplest possible baseline. Its MSE tells us how much non-linear structure exists in the data.

**What changed in `train.py`:** Initial version — `LinearRegression().fit(train_X, train_y)`.

**Result:** val_mse = 76.041106

**Conclusion:** The very high MSE confirms the data is dominated by non-linear structure (primarily `x1²`) that a linear model cannot capture at all.

**Next direction:** Add polynomial features (degree 2) to let the linear model capture quadratic and interaction terms.
```

### 7.5 Persistence

`experiment_log.md` is **gitignored** — like `results.tsv`, it persists across `git reset --hard` calls. The agent's full reasoning history survives discards.

### 7.6 File Management

Add to `.gitignore`:

```
results.tsv
experiment_log.md
run.log
```

---

## 8. Git Workflow

### 8.1 Branch

```bash
git checkout -b agentic-ml/run1
```

Work happens entirely on this branch. `main` stays clean.

### 8.2 Commit Format

Short imperative sentence describing the hypothesis:

```
add polynomial degree-2 features
switch to GradientBoostingRegressor
add explicit sin(x2) feature
ensemble linear + GBM predictions
```

### 8.3 Discard Mechanics

On discard, undo the last commit but keep the working tree clean:

```bash
git reset --hard HEAD~1
```

This is safe because:
- We are on a non-protected branch (`agentic-ml/run1`)
- `results.tsv` is gitignored and unaffected
- The commit being removed was made in this session and has not been pushed

### 8.4 Keep/Discard Decision Point

The decision is made **after** parsing `run.log`. The commit already exists on the branch before the decision. On discard, it is removed retroactively. This matches autoresearch's approach exactly.

---

## 9. Time Budget and Timeouts

| Parameter | Value | Rationale |
|---|---|---|
| Expected runtime per experiment | < 5 seconds | 100 rows, scikit-learn |
| Soft budget (warn if exceeded) | 30 seconds | Generous for complex ensembles |
| Hard kill timeout | 60 seconds | Double the soft budget |
| Timeout treatment | CRASH | Log `val_mse = inf`, discard |

The agent enforces the timeout via its own monitoring when running `evaluate.py`. If a subprocess approach is used, `subprocess.run(..., timeout=60)` is the mechanism.

---

## 10. Dependency Installation

Before the first run:

```bash
uv add numpy pandas scikit-learn scipy xgboost lightgbm
```

Update `pyproject.toml` with these dependencies. Python 3.13+ as specified.

---

## 11. Expected Trajectory

| Stage | Approximate val_mse | Likely discovery |
|---|---|---|
| Baseline | ~76 | Linear regression misses `x1²` which dominates variance |
| Early improvement | ~5–20 | Polynomial features or tree model captures `x1²` |
| Mid improvement | ~1–5 | GBM or explicit interaction `x1·x2` features |
| Late improvement | ~0.3–1.0 | Tuned ensemble or explicit `sin(x2)` feature |
| Near-optimal | ~0.25–0.4 | Approaching noise floor |

The noise floor (~0.25) is the theoretical minimum achievable MSE (σ² = 0.5²). An agent that discovers and explicitly encodes `x1²`, `sin(x2)`, and `x1·x2` as features will approach this limit.

---

## 12. Implementation Checklist

- [ ] Add `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm` to `pyproject.toml`
- [ ] Add `results.tsv`, `experiment_log.md`, and `run.log` to `.gitignore`
- [ ] Create `evaluate.py` exactly as specified in Section 3
- [ ] Create `train.py` with baseline linear regression (Section 4.2)
- [ ] Create `program.md` with loop instructions (Section 5)
- [ ] Create branch `agentic-ml/run1`
- [ ] Run `python evaluate.py` manually to verify the harness works and record the baseline MSE
- [ ] Initialize `results.tsv` with header row and baseline entry
- [ ] Initialize `experiment_log.md` with header block and baseline entry
- [ ] Commit baseline `train.py` as the first entry
- [ ] Begin the agent loop
