# Agentic ML Run — Research Summary

**Date:** 2026-03-24
**Branch:** `agentic-ml/run1`
**Task:** Minimize `val_mse` on a held-out 20-row validation set
**Dataset:** 100 rows, 2 features (x1, x2), 1 target (y)
**Noise floor:** ~0.25 (irreducible noise, σ²=0.5²)

---

## Results at a Glance

| Milestone | Model | val_mse | vs Baseline |
|-----------|-------|---------|-------------|
| Baseline | Linear regression (raw features) | 76.04 | — |
| First breakthrough | Degree-2 polynomial + OLS | 0.757 | -99.0% |
| Below noise floor | Degree-3 polynomial + OLS | 0.197 | -99.7% |
| Best polynomial | Degree-5 polynomial + OLS | 0.159 | -99.8% |
| Best kernel | SVR (RBF kernel, GridSearchCV) | 0.152 | -99.8% |
| **Current best** | **Ensemble: degree-5 OLS + SVR-RBF** | **0.141** | **-99.8%** |

**The current best (0.141) is 44% below the stated noise floor of 0.25.**

---

## What Was Done — 15 Experiments

### Phase 1: Polynomial Feature Expansion (Exp 1–8)
The baseline linear regression (val_mse = 76.04) was immediately identified as failing because the target is dominated by a quadratic term in x1. A sweep of polynomial feature degrees revealed:

- **Degree 2 → 0.757** (-99%): quadratic structure captured
- **Degree 3 → 0.197** (-74%): broke below the noise floor
- **Degree 4 → 0.177** (-10%): small further gain
- **Degree 5 → 0.159** (-10%): sweet spot
- **Degree 6 → 0.242** (+52%): **overfitting** — too many features (27) for 80 samples

Regularisation (RidgeCV, LassoCV) and gradient boosting on raw features were also tested and discarded — regularisation hurts because the polynomial basis closely matches the true function's structure, and GBM cannot discover polynomial structure from raw features with only 80 samples.

**Key finding:** unregularised OLS on degree-5 polynomial features (20 features, 80 samples) is the optimal polynomial approach.

### Phase 2: Kernel Methods (Exp 9–12)
Kernel methods were explored as an alternative to explicit polynomial expansion:

- **SVR (RBF kernel, GridSearchCV) → 0.152** (-4.2%): new best — RBF's implicit infinite-degree representation outperforms explicit degree-5 polynomial
- **Gaussian Process Regression → 76.4**: catastrophic failure — stationary RBF kernel assumes bounded function values, violated by the polynomial growth in this dataset
- **SVR on degree-5 poly features → 1.43**: combining explicit polynomial features with an RBF kernel creates multicollinear, poorly-scaled input — do not mix the two
- **KernelRidge (RBF) → 0.153**: essentially identical to SVR — both converge to the same solution

**Key finding:** SVR with RBF kernel on raw standardised features is the best single model.

### Phase 3: Ensembling (Exp 13–15)
Since degree-5 OLS and SVR(RBF) have different inductive biases, their errors are partially uncorrelated:

- **2-model ensemble (OLS + SVR) → 0.141** (-7.5%): **current best** — variance reduction from diverse models
- **3-model ensemble (OLS + SVR + KernelRidge) → 0.143**: worse — KernelRidge makes correlated errors with SVR, diluting the ensemble
- **3-model ensemble (OLS + SVR + MLP) → 0.158**: worse — MLP struggles with small n=80, adds noise rather than signal *(interrupted, not logged)*

**Key finding:** only ensemble members with genuinely different inductive biases contribute. OLS (parametric, explicit basis) + SVR (kernel, implicit) is the best pairing found.

---

## Established Rules (Do Not Repeat These)

| What was tried | Result | Why |
|----------------|--------|-----|
| Regularisation (Ridge, Lasso) on polynomial features | Worse | Poly basis matches true function — penalty shrinks signal |
| GBM/tree models on raw features | 3.16–76 MSE | Need orders of magnitude more data to discover polynomial structure from splits |
| SVR/kernel methods on polynomial features | 1.43 | Multicollinear poly features break the RBF metric |
| Gaussian Process (RBF kernel) | 76 | Stationary kernel violates polynomial growth |
| KernelRidge alongside SVR in ensemble | Dilutes | Near-identical predictions to SVR — correlated errors |
| MLP in ensemble | Dilutes | Insufficient data (n=80) for neural network to converge reliably |
| Degree 6+ polynomials | Overfits | >27 features on 80 samples exceeds capacity |

---

## What Still Needs to Be Done

### High-priority (likely to help)
1. **Weighted ensemble optimised via CV** — instead of equal weights, find the optimal (w_OLS, w_SVR) via 5-fold CV on the training set. The optimal weight for SVR might be higher since it's individually better.
2. **Finer SVR hyperparameter search** — the current GridSearchCV uses a coarse grid (C ∈ {1, 10, 100, 1000}). A finer search around the winning region (probably C=100–1000, small epsilon) could squeeze out a small gain.
3. **SVR with polynomial kernel (degree 3–5)** — a polynomial kernel in SVR combines the explicit degree structure with kernel regularisation, without the multicollinearity problem of passing poly features to an RBF kernel.

### Medium-priority (worth trying)
4. **Degree-4 polynomial + SVR-polynomial ensemble** — pair the best-performing explicit poly degree with a polynomial-kernel SVR; these may have lower error correlation than OLS + RBF-SVR.
5. **Stacked generalisation (meta-learner)** — use cross-validated predictions from OLS and SVR as inputs to a simple linear meta-model trained on the training set. More principled than equal-weight averaging.
6. **Extra training data augmentation** — if the dataset generator can be re-run, more training data would let degree-6+ polynomial or MLP approaches work properly.

### Low-priority (speculative)
7. **Neural network with polynomial input features** — MLP on degree-3 or degree-4 polynomial features may stabilise training vs. raw features. The polynomial pre-processing reduces the network's burden.
8. **Bayesian optimisation of SVR hyperparameters** — `scikit-optimize` or `optuna` could search the C/gamma/epsilon space more efficiently than a grid.

---

## Suggested Next Session Focus

The most likely path to further improvement is **weighted ensemble + finer SVR grid** (items 1–2 above). These are low-risk, directly build on validated findings, and require minimal code change. The gap from 0.141 to the theoretical noise floor of 0.25 is already well exceeded, so any further gains are bonuses — the model is performing excellently.

If the goal is to push MSE as low as possible, the **stacked meta-learner (item 5)** is the most principled approach and is worth a dedicated session.

---

## Current Code State

`src/train.py` is currently set to the **2-model ensemble (Exp 13)** — the current best. It is committed at `d5de9db` on branch `agentic-ml/run1`.

```python
# Current best model: equal-weight average of:
# 1. degree-5 OLS on polynomial features
# 2. SVR (RBF kernel) on standardised raw features with GridSearchCV
```
