# PLAN.md — Agentic ML Learning Loop

## 1. What This Project Is

An autonomous ML improvement loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI coding agent (Claude Code) is given a dataset, a metric to optimize, and a model training file. It then enters a loop: hypothesize a change → edit the training code → run the experiment → evaluate the metric → keep or discard. This continues indefinitely without human intervention, producing a monotonically improving chain of experiments.

**Autoresearch trains a GPT language model on text data and optimizes val_bpb.** Our project generalizes this: given *any* tabular dataset with a computable evaluation metric, the agent iterates on model architecture, features, and hyperparameters to drive that metric as low (or high) as possible.

---

## 2. How Autoresearch Works (and How We Adapt It)

Autoresearch's design rests on a strict separation of concerns across three files:

| Autoresearch file | Role | Our equivalent |
|---|---|---|
| `prepare.py` | Fixed infrastructure: data loading, tokenization, evaluation harness. **The agent cannot touch this.** | `evaluate.py` — loads the dataset, splits train/val, and computes the target metric. Read-only to the agent. |
| `train.py` | Model definition, optimizer, training loop. **The agent edits only this file.** | `train.py` — model definition, feature engineering, hyperparameters, training loop. The agent's single point of control. |
| `program.md` | Natural-language instructions that define the agent's behavior: loop rules, logging format, keep/discard logic, constraints. | `program.md` — same role, adapted to our tabular ML context. |

The critical insight from autoresearch: **the human programs the agent's behavior (in program.md), the agent programs the model (in train.py), and the evaluation harness (evaluate.py) is a trusted, fixed contract between them.** This three-way separation is what makes autonomous operation safe — the agent can never game the metric by modifying the evaluator.

### The Loop (from autoresearch, adapted)

```
LOOP FOREVER:
  1. Read current git state and results history (results.tsv)
  2. Formulate a hypothesis (e.g., "add polynomial features", "switch to gradient boosting")
  3. Edit train.py to implement the hypothesis
  4. Git commit the change with a descriptive message
  5. Run: python train.py > run.log 2>&1 (with a fixed time budget)
  6. Extract results: grep for the metric line in run.log
  7. If no metric found → crash. Read the log tail, attempt a fix or skip.
  8. Append results to results.tsv
  9. If metric improved → KEEP (advance the branch)
 10. If metric same or worse → DISCARD (git reset --hard to previous commit)
```

---

## 3. Our Domain: Tabular Regression

Instead of training a language model, our agent optimizes a regression model on a small tabular dataset.

**Dataset:** 100 rows, 2 input variables (`x1`, `x2`), 1 target variable (`y`). The relationship has a non-trivial structure (non-linear with interaction effects and noise) so that the agent has room to discover improvements — a simple linear model won't be optimal, but the true pattern is discoverable.

**Metric:** Mean Squared Error (MSE) on a held-out validation set. Lower is better. The evaluation harness uses a fixed random seed for the train/val split so results are deterministic and comparable across experiments.

**Why this is a good testbed:**
- Small enough to train in seconds (no GPU required), so the agent can iterate quickly
- Complex enough that naive approaches leave room for improvement
- The metric is unambiguous and cheap to compute
- The agent can explore feature engineering, model selection, hyperparameter tuning, and ensembling — all the same categories of changes that autoresearch's agent makes, but in a more accessible domain

---

## 4. File Architecture — What the Technical Spec Must Define

### 4.1 `evaluate.py` (Fixed Infrastructure — Agent Cannot Edit)

This is the trusted evaluation contract. It must define:

**Data loading:**
- Read the CSV dataset from a fixed path (`data/dataset.csv`)
- Split into train and validation sets using a fixed random seed (e.g., 80/20 split, `seed=42`)
- The split must be identical every run so that metric comparisons are valid

**Evaluation function:**
- `evaluate(predictions, val_y) -> float` — computes MSE between the model's predictions on the validation set and the true values
- Must handle edge cases: NaN predictions, infinite values, wrong-length arrays — all should return a large penalty value (e.g., `float('inf')`) rather than crashing

**The main evaluation entry point:**
- Imports the agent's `train.py` module
- Calls a fixed interface: `train.py` must expose a function `train_and_predict(train_X, train_y, val_X) -> val_predictions`
- Computes and prints the metric in a parseable format: `val_mse: 0.042317`
- Also prints metadata: `num_rows_train`, `num_rows_val`, `elapsed_seconds`

**Why this contract matters:** The agent can do anything inside `train_and_predict` — use scikit-learn, write raw numpy, build an ensemble — but it must accept the exact inputs and return predictions in the exact shape. This is the equivalent of autoresearch's `evaluate_bpb()`: a fixed, trusted measurement that the agent's creativity operates within.

### 4.2 `train.py` (Agent-Editable — The Experiment Surface)

This is the *only* file the agent modifies. The spec must define:

**Required interface:**
- Must expose `train_and_predict(train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray) -> np.ndarray`
- `train_X` shape: `(n_train, 2)`, `val_X` shape: `(n_val, 2)`, return shape: `(n_val,)`

**Baseline implementation:**
- The spec should define a simple starting point (e.g., `sklearn.linear_model.LinearRegression`) so the agent has a known baseline to improve from
- The baseline establishes the first row of `results.tsv` and the first commit on the experiment branch

**What the agent is free to change:**
- Model type (linear regression → random forest → gradient boosting → neural network → ensemble)
- Feature engineering (polynomial features, interactions, transformations, binning)
- Hyperparameters (learning rate, tree depth, regularization strength)
- Training procedure (cross-validation for hyperparameter selection, early stopping)
- Anything, as long as the function signature is preserved

**What the agent must NOT do:**
- Read or modify `evaluate.py` or the dataset file
- Hardcode validation set answers
- Import or read data outside of the function arguments
- Make network requests

### 4.3 `program.md` (Agent Instructions)

This is the "program" for the agent itself — written in natural language, executed by Claude Code. The spec must define:

**Setup protocol:**
- How the agent initializes: check git state, run the baseline, record the first result
- How to create the experiment branch (e.g., `agentic-ml/run1`)

**Loop rules:**
- Time budget per experiment (e.g., 60 seconds max — our dataset is tiny)
- Kill timeout (e.g., 120 seconds — double the budget, to catch hangs)
- How to handle crashes: read the log, fix trivial bugs, re-run once; if still broken, discard and move on
- How to handle ties: treat as discard (no improvement = no value)

**Logging format:**
- Structured output at end of each run: `val_mse: <float>`, `elapsed_seconds: <float>`
- `results.tsv` format: `commit | val_mse | status | description`
- The agent appends to `results.tsv` after every experiment (kept untracked in git)

**Keep/discard logic:**
- If `val_mse` decreased → KEEP: the commit stays, the branch advances
- If `val_mse` stayed same or increased → DISCARD: `git reset --hard` to the previous commit
- If the run crashed → CRASH: log with `val_mse = Inf`, discard

**Strategy guidance (from autoresearch's philosophy):**
- Favor simplicity: a small improvement that adds significant complexity is not worth it
- Value deletions: if removing code maintains performance, that's an improvement
- Think before coding: formulate a clear hypothesis before editing
- Learn from history: read `results.tsv` to avoid repeating failed approaches
- Explore broadly first, then exploit: don't tunnel-vision on one approach

### 4.4 `data/dataset.csv` (Fixed Dataset)

The spec must define:
- 100 rows, columns: `x1`, `x2`, `y`
- A ground-truth generating function with enough complexity to reward feature engineering (e.g., interaction terms, non-linearity) but not so much complexity that it's pure noise
- A fixed noise component so the agent can approach but never perfectly achieve MSE = 0
- Delivered as a CSV file, committed to the repo, never modified

### 4.5 `run.py` (Loop Orchestrator — Optional)

Autoresearch relies on the AI agent itself to execute the loop (the agent runs `uv run train.py`, reads the log, decides keep/discard). This is the simplest approach and is what `program.md` describes.

However, the spec may optionally define a Python orchestrator that:
- Runs `evaluate.py` with a subprocess timeout
- Parses the output
- Appends to `results.tsv`
- Executes the git keep/discard logic

**Tradeoff:** An orchestrator makes the loop more robust (handles timeouts reliably, standardizes logging) but reduces the agent's autonomy. Autoresearch chose no orchestrator — the agent handles everything. The spec should decide which approach to take.

---

## 5. Technical Details the Spec Must Flush Out

These are the specific decisions and definitions that SPEC.md needs to resolve:

### 5.1 The Evaluation Contract

- **Exact function signature** for `train_and_predict` — input types, shapes, return type
- **Error handling policy** — what happens if the function raises an exception, returns wrong shapes, or returns NaN values? The evaluator needs explicit behavior for each case.
- **Metric precision** — how many decimal places to compare? Autoresearch uses 6 (e.g., `0.997900`). We should do the same to distinguish small improvements.
- **Determinism guarantee** — the evaluator must produce identical results for identical `train.py` code. This means: fixed random seed in the split, and the spec should mandate that `train.py` also sets its own random seeds for reproducibility.

### 5.2 The Agent's Allowed Dependencies

- What packages can `train.py` import? Autoresearch constrains this implicitly (only PyTorch + custom kernels). We need an explicit list.
- Recommended: `numpy`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm` — installed in the environment before the loop starts
- Forbidden: anything that makes network requests, reads files outside the working directory, or spawns subprocesses

### 5.3 The Dataset Generating Function

- The spec must define the exact function used to generate `y` from `x1` and `x2`
- This function is NOT revealed to the agent — it only sees the CSV
- The function should have:
  - A non-linear component (e.g., `sin`, `x^2`, `exp`)
  - An interaction term (e.g., `x1 * x2`)
  - Gaussian noise with a known variance (this sets the floor for achievable MSE)
- Example structure (not final): `y = 3*x1^2 + 2*sin(x2) + 1.5*x1*x2 + noise`
- The spec must record the function and noise level so we can compute the theoretical minimum MSE

### 5.4 Git Workflow

- **Branch naming**: `agentic-ml/<run-name>` (e.g., `agentic-ml/run1`)
- **Commit message format**: short description of the hypothesis (e.g., "add polynomial degree-2 features")
- **Keep/discard mechanics**: `git reset --hard HEAD~1` on discard — the spec must confirm this is safe given the branch structure
- **results.tsv**: gitignored, so it persists across resets. The spec must include `.gitignore` rules.

### 5.5 Time Budget and Timeout

- **Per-experiment wall-clock budget**: the max expected runtime. For a 100-row dataset with scikit-learn, even complex models should finish in under 10 seconds. Suggest 30-second budget, 60-second kill timeout.
- **Timeout mechanism**: `subprocess.run(timeout=...)` if using an orchestrator, or agent-managed via `program.md` instructions
- **What "timeout" means**: treat as a crash, log it, discard

### 5.6 Logging and Structured Output

The spec must define the exact parseable output format printed by `evaluate.py`:

```
val_mse:          0.042317
elapsed_seconds:  1.3
num_rows_train:   80
num_rows_val:     20
```

And the exact `results.tsv` format:

```
commit    val_mse      status    description
a1b2c3d   0.042317     keep      baseline linear regression
b2c3d4e   0.038910     keep      add polynomial degree-2 features
c3d4e5f   0.045200     discard   switch to decision tree (overfit)
```

### 5.7 The Starting Point

- The spec must provide the complete baseline `train.py` (a simple linear regression)
- The baseline's expected MSE should be documented so we know the starting point
- The agent's first action is to run the baseline, confirm the MSE matches, and commit it as the first entry in `results.tsv`

---

## 6. What Success Looks Like

After running overnight (or for an hour — our experiments are fast):

1. **results.tsv** contains 50–200+ experiments, showing the agent's exploration trajectory
2. The experiment branch has a clean chain of kept commits, each an improvement
3. The final `val_mse` is meaningfully lower than the baseline
4. The agent discovered non-trivial improvements: feature engineering, model selection, hyperparameter tuning, or ensembling — not just random guessing

The value of this project is not the final model — it's the **framework**: a reusable system where you can swap in any dataset and metric, and the agent will autonomously iterate toward a better solution.

---

## 7. Next Step

Create **SPEC.md**: a precise technical specification that resolves every open question above and defines the exact file contents, interfaces, and behavior so that the implementation can proceed without ambiguity.
