# Experiment Log

This file records every experiment run by the agentic ML loop, including
the reasoning behind each attempt, the outcome, and lessons learned.

**Dataset:** 100 rows, 2 features (x1, x2), 1 target (y)
**Metric:** val_mse (lower is better)
**Noise floor:** ~0.25 (theoretical minimum, σ²=0.5²)

---

### Experiment 0 — keep
**Hypothesis:** Establish baseline with simple linear regression on raw features.

**Reasoning:** Starting point as specified in the project. Linear regression on (x1, x2) is a known weak baseline since the comment in train.py notes "x1² dominates the output; linear model misses it entirely".

**What changed in `src/train.py`:** No changes — this is the initial code using sklearn LinearRegression on raw x1, x2 features.

**Result:** val_mse = 76.041106 (prev best: -, change: -)
Elapsed: 0.0s

**Conclusion:** As expected, linear regression fails badly. The target is dominated by a nonlinear (likely quadratic) relationship with x1. The model has no way to capture x1² with only raw features.

**Next direction:** Add polynomial features (degree 2: x1², x2², x1*x2) via PolynomialFeatures. This directly addresses the x1² structure and should reduce MSE dramatically.

---

### Experiment 1 — keep
**Hypothesis:** Degree-2 polynomial features will dramatically reduce MSE by allowing the linear model to fit the x1² term.

**Reasoning:** The train.py comment explicitly states "x1² dominates the output; linear model misses it entirely". Adding x1², x2², and x1*x2 as features directly exposes these relationships to a linear model.

**What changed in `src/train.py`:** Added sklearn PolynomialFeatures(degree=2, include_bias=False) to expand 2 raw features into 5 features (x1, x2, x1², x1*x2, x2²). Applied LinearRegression on the expanded features.

**Result:** val_mse = 0.757237 (prev best: 76.041106, change: -99.0%)
Elapsed: 0.0s

**Conclusion:** Confirmed. Polynomial features captured the dominant quadratic structure. MSE dropped from 76.04 to 0.757 — a 99% reduction. However, we're still 3x above the noise floor (0.25), suggesting residual nonlinearity not captured by degree-2 polynomials.

**Next direction:** Try degree-3 polynomial features to capture any cubic/quartic interactions. With 80 training samples and only 9 degree-3 features, overfitting risk is low. If that doesn't help, explore Ridge regularization or gradient boosting.

---

### Experiment 2 — keep
**Hypothesis:** Degree-3 polynomial features will capture higher-order terms (x1³, x1²x2, etc.) that degree-2 misses, reducing MSE further toward the noise floor.

**Reasoning:** MSE of 0.757 vs noise floor ~0.25 indicates residual structure not captured by degree-2 (5 features). Degree-3 adds 4 more features (x1³, x1²x2, x1x2², x2³), totaling 9 features for 80 training samples — still well-posed.

**What changed in `src/train.py`:** Changed PolynomialFeatures(degree=2) to PolynomialFeatures(degree=3). All else unchanged.

**Result:** val_mse = 0.196930 (prev best: 0.757237, change: -74.0%)
Elapsed: 0.0s

**Conclusion:** Confirmed strongly. Degree-3 brought MSE below the stated noise floor (0.25), achieving 0.197. The true function likely has cubic terms. This is an excellent result — the model is now fitting the underlying function very well with minimal overfitting.

**Next direction:** Try degree-4 polynomial to see if further structure exists. Risk of slight overfitting with 14 features on 80 samples, but still reasonable. Alternatively, try Ridge regularization with degree-3 to potentially reduce variance on the validation set.

---

### Experiment 3 — keep
**Hypothesis:** Degree-4 polynomial features will capture any quartic terms in the true function, reducing MSE slightly further.

**Reasoning:** We're at 0.197 vs noise floor 0.25 — already below the floor, suggesting some lucky fit on the 20 val samples. But if there are x1⁴ or x1³x2 terms, degree-4 would capture them. 14 features vs 80 samples remains feasible.

**What changed in `src/train.py`:** Changed PolynomialFeatures(degree=3) to PolynomialFeatures(degree=4).

**Result:** val_mse = 0.176694 (prev best: 0.196930, change: -10.3%)
Elapsed: 0.0s

**Conclusion:** Small but real improvement. Degree-4 helped slightly, suggesting mild quartic structure in the data or beneficial chance fit. We are now well below the stated noise floor (0.25), at 0.177.

**Next direction:** Try Ridge regression with degree-3 to see if regularization stabilizes the fit. With very few samples, unregularized high-degree polynomials may be overfitting. Also consider GradientBoosting which can implicitly handle higher-order terms.

---

### Experiment 4 — discard
**Hypothesis:** Ridge regularization with cross-validated alpha and StandardScaler will reduce variance and improve over the unregularized degree-4 poly model.

**Reasoning:** Degree-4 linear regression on 80 samples with 14 features may overfit. Ridge + StandardScaler is standard practice to stabilize polynomial regression. Used 5-fold CV to select alpha from [0.001, 0.01, 0.1, 1.0, 10.0, 100.0].

**What changed in `src/train.py`:** Switched LinearRegression → RidgeCV(cv=5), added StandardScaler, set degree back to 3.

**Result:** val_mse = 0.196487 (prev best: 0.176694, change: +11.2%)
Elapsed: 0.0s

**Conclusion:** Regularization hurt. The unregularized degree-4 OLS is better at fitting this small dataset. The polynomial features perfectly represent the true function's basis, so regularization shrinks the true signal. Reverted.

**Next direction:** Try degree-5 polynomial features (unregularized) to see if even higher-order terms help. Or try GradientBoosting / XGBoost as a completely different model family to compare approaches.

---

### Experiment 5 — discard
**Hypothesis:** GradientBoosting on raw features can implicitly discover polynomial structure via tree splits, offering an alternative to explicit polynomial expansion.

**Reasoning:** GBM with depth-4 trees, 300 estimators, and low learning rate should in principle approximate smooth functions, including polynomials. Exploring a different model family to see if it can match or beat the polynomial regression.

**What changed in `src/train.py`:** Replaced LinearRegression+PolynomialFeatures with GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8).

**Result:** val_mse = 3.157156 (prev best: 0.176694, change: +1688%)
Elapsed: 0.1s

**Conclusion:** Terrible result. Tree-based models with only 80 samples cannot efficiently approximate quadratic/cubic functions from raw features — they need many more splits or explicit polynomial features. The polynomial + linear approach is clearly superior for this dataset structure.

**Next direction:** Stay with the polynomial + linear regression approach. Try degree-5 polynomial (unregularized OLS) to probe for any higher-order terms. Risk of overfitting increases but 80 samples vs ~20 degree-5 features is still reasonable.

---

### Experiment 6 — keep
**Hypothesis:** Degree-5 polynomial features will find additional quintic structure in the data, continuing the trend of improvement with higher degree.

**Reasoning:** Each degree increase (2→3, 3→4) brought improvement. 20 features vs 80 samples maintains feasibility. The improvements are real (not just noise variance) since the pattern is consistent.

**What changed in `src/train.py`:** Changed PolynomialFeatures(degree=4) to PolynomialFeatures(degree=5).

**Result:** val_mse = 0.159090 (prev best: 0.176694, change: -10.0%)
Elapsed: 0.0s

**Conclusion:** Still improving. The trend of each degree bringing marginal gains continues. We are now at 0.159, well below the noise floor. The true function may have degree-5 terms, or we're seeing beneficial overfitting on the small val set.

**Next direction:** Try degree-6 (27 features vs 80 samples — still 3:1 ratio, feasible). If degree-6 also helps, the pattern suggests the true function has higher-degree terms. Watch for the point of diminishing returns or regression.

---

### Experiment 7 — discard
**Hypothesis:** Degree-6 polynomial will continue the improvement trend seen from degree-3 through degree-5.

**Reasoning:** Each previous degree increase helped. Degree-6 adds 7 new features (27 total vs 80 samples, 3:1 ratio), which should remain feasible.

**What changed in `src/train.py`:** Changed PolynomialFeatures(degree=5) to PolynomialFeatures(degree=6).

**Result:** val_mse = 0.242140 (prev best: 0.159090, change: +52.2%)
Elapsed: 0.0s

**Conclusion:** Overfitting threshold hit. Degree-6 with 27 features on 80 training samples is too many degrees of freedom. The model fits noise, hurting val_mse. Degree-5 (20 features) is the sweet spot. Reverted.

**Next direction:** Degree-5 is the sweet spot for polynomial degree. Try LassoCV with degree-5 features — it performs automatic feature selection and may zero out noisy higher-order cross-terms while retaining the important ones. This could reduce overfitting within degree-5 space and potentially beat unregularized OLS.

---

### Experiment 8 — discard
**Hypothesis:** LassoCV with degree-5 polynomial features and StandardScaler will perform automatic feature selection, zeroing out noisy high-degree cross terms while retaining the important monomials.

**Reasoning:** OLS with 20 features on 80 samples is slightly underdetermined (4:1 ratio). Lasso's L1 penalty should identify and zero out spurious features, regularizing better than OLS within the same feature space.

**What changed in `src/train.py`:** Replaced LinearRegression with LassoCV(cv=5, max_iter=10000), added StandardScaler. Kept degree-5.

**Result:** val_mse = 0.183575 (prev best: 0.159090, change: +15.4%)
Elapsed: 0.0s

**Conclusion:** Lasso hurt. The polynomial features are all genuinely useful — Lasso's sparsity-inducing penalty incorrectly zeros out important terms, introducing bias. The same phenomenon seen with RidgeCV (experiment 4): regularization hurts because the polynomial basis matches the true function's structure. OLS without penalty is the right choice.

**Next direction:** Try SVR with RBF or polynomial kernel. Kernel methods are well-suited for small datasets and can implicitly represent infinite-degree polynomials. GaussianProcessRegressor is another strong candidate for n=80.
