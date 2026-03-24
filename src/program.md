# program.md — Agentic ML Loop Instructions

You are an autonomous ML optimization agent. Your goal: minimize `val_mse` on
the validation set by iteratively improving `src/train.py`.

You maintain two log files that together provide a complete record of your work:
- `results.tsv` — structured data: every experiment's number, commit, metric,
  comparison to previous best, timing, status, model type, and description.
- `experiment_log.md` — narrative journal: your reasoning, hypothesis, what you
  changed, what happened, what you learned, and what you plan to try next.

These logs are the primary deliverable of your work — a human should be able to
read `experiment_log.md` and fully understand your decision-making process.

---

## Setup (run once at the start)

1. Ensure you are on branch `agentic-ml/run1` (create it from main if needed):
   ```
   git checkout -b agentic-ml/run1
   ```

2. Initialize `results.tsv` with the header row (tab-separated):
   ```
   experiment	commit	val_mse	prev_best_mse	improvement	elapsed_sec	status	model_type	description
   ```

3. Initialize `experiment_log.md` with the header block:
   ```
   # Experiment Log

   This file records every experiment run by the agentic ML loop, including
   the reasoning behind each attempt, the outcome, and lessons learned.

   **Dataset:** 100 rows, 2 features (x1, x2), 1 target (y)
   **Metric:** val_mse (lower is better)
   **Noise floor:** ~0.25 (theoretical minimum, σ²=0.5²)
   ```

4. Run the baseline:
   ```
   python src/evaluate.py > run.log 2>&1
   ```

5. Read `run.log`. Extract `val_mse` and `elapsed_seconds`.

6. Record the baseline in both log files (see logging format below).

7. Print the baseline MSE and begin the loop.

---

## The Loop

Repeat forever until manually stopped.

### Step 1: Hypothesize

- Read `experiment_log.md` — especially the **Next direction** from the last entry.
- Read `results.tsv` to see what has been tried and what worked.
- Look at the current `src/train.py`.
- Form a specific, testable hypothesis. Write it down mentally — you'll record it in the log.
- Do not repeat approaches that have already been discarded.

### Step 2: Implement

- Edit `src/train.py` **only**. Do not touch any other file.
- Keep the function signature `train_and_predict(train_X, train_y, val_X)` intact.
- Set random seeds inside your code for reproducibility.

### Step 3: Commit

```
git add src/train.py
git commit -m "<short description of the hypothesis>"
```

### Step 4: Run

```
python src/evaluate.py > run.log 2>&1
```

Time budget: 30 seconds. If the process has not finished in 60 seconds, kill it
and treat it as a CRASH.

### Step 5: Parse Result

- Read `run.log`.
- Find the line starting with `val_mse:` and extract the float.
- Find `elapsed_seconds:` and extract the float.
- If no `val_mse:` line exists → CRASH.

### Step 6: Keep or Discard

Let `prev_mse` = the lowest `val_mse` among all rows with `status = keep` in `results.tsv`.

- **KEEP** if `new_mse < prev_mse`:
  - The commit stays on the branch.
  - Go to Step 7 with `status = keep`.

- **DISCARD** if `new_mse >= prev_mse`:
  - `git reset --hard HEAD~1`
  - Go to Step 7 with `status = discard`.

- **CRASH** (no `val_mse:` line, or exception in run.log):
  - `git reset --hard HEAD~1`
  - Read the tail of `run.log` to understand the error.
  - Do not retry the same broken approach.
  - Go to Step 7 with `status = crash` and `val_mse = inf`.

### Step 7: Log (mandatory after every experiment)

**7a. Append one row to `results.tsv`** (tab-separated, 9 columns):

```
<N>	<commit>	<val_mse>	<prev_best>	<improvement%>	<elapsed_sec>	<status>	<model_type>	<description>
```

Where:
- `<N>` is the experiment number (increment by 1 each time, never resets)
- `<commit>` is the 7-character short git hash
- `<improvement%>` is the percentage change from `prev_best` (e.g. `-34.2%` or `+9.1%` or `-` for baseline/crash)
- `<model_type>` is the primary sklearn/xgboost class name used

**7b. Append one entry to `experiment_log.md`** — all six sections are required:

```
---

### Experiment <N> — <status>
**Hypothesis:** <one sentence: what you are testing>

**Reasoning:** <why you expect this to help, citing evidence from prior experiments>

**What changed in `src/train.py`:** <plain-English description of the code changes>

**Result:** val_mse = <value> (prev best: <value>, change: <improvement%>)
Elapsed: <seconds>s

**Conclusion:** <what the result tells you — did it confirm or refute the hypothesis? why?>

**Next direction:** <what you plan to try next, based on what you just learned>
```

Do not skip or abbreviate any section. The log is the primary record of your thinking.

**Return to Step 1.**

---

## Strategy

- **Think before coding.** Know *why* a change should help before writing it.
- **Learn from history.** The "Next direction" from your last log entry is your starting point each iteration.
- **Explore before exploiting.** Try different model families early (linear, tree, boosting, neural). Tune once you've found a promising direction.
- **Favor simplicity.** A small gain with clean code beats a large gain with fragile code.
- **Value deletions.** Removing code that doesn't help is an improvement.
- **The noise floor is ~0.25.** You cannot go below this. Below ~0.4 is excellent.

---

## Constraints

- Never edit `src/evaluate.py`, `data/dataset.csv`, or any file other than `src/train.py`.
- Never hardcode validation answers.
- Never read files or make network requests inside `train_and_predict`.
- Always write the full log entry — no skipping.
- **Never stop to ask the human.** The human may be away. Run until manually interrupted.
