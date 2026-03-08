import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({"figure.dpi": 110, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


# ══════════════════════════════════════════════════════════════════
# PART A — DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("  PART A — Dataset Construction")
print("=" * 60)

N       = 10000
SEED    = 42
rng     = np.random.RandomState(SEED)

# ── A1: Clean Dataset  y = ax + b + ε  ───────────────────────────
a_true, b_true, sigma_noise = 3.5, 1.2, 0.8
x_clean = rng.uniform(-5, 5, N)
eps     = rng.normal(0, sigma_noise, N)
y_clean = a_true * x_clean + b_true + eps

X_clean = np.column_stack([np.ones(N), x_clean])   # design matrix [1, x]

print(f"\nA1 Clean Dataset:")
print(f"  y = {a_true}x + {b_true} + ε,  ε ~ N(0, {sigma_noise}²)")
print(f"  x ~ Uniform(-5, 5),  N = {N:,},  seed = {SEED}")

# ── A2: Correlated Feature Dataset ───────────────────────────────
a_multi = np.array([2.0, -1.5, 0.8])   # true coefficients for [x1, x2, x3]
x1 = rng.uniform(-5, 5, N)
x2 = 2.5 * x1 + rng.normal(0, 0.15, N)    # nearly linear in x1
x3 = rng.normal(0, 1, N)
X_corr = np.column_stack([np.ones(N), x1, x2, x3])
y_corr = X_corr[:, 1:] @ a_multi + rng.normal(0, 0.5, N)

XtX_corr = X_corr.T @ X_corr
eigvals_corr = np.linalg.eigvalsh(XtX_corr)
cond_corr    = eigvals_corr.max() / eigvals_corr.min()

print(f"\nA2 Correlated Feature Dataset:")
print(f"  x2 = 2.5*x1 + N(0, 0.15),  x3 ~ N(0,1),  N = {N:,}")
print(f"  Eigenvalues of XᵀX : {np.sort(eigvals_corr)[::-1].round(1)}")
print(f"  Condition number    : {cond_corr:.2e}")

# ── A3: Outlier Dataset ───────────────────────────────────────────
outlier_frac = 0.12    # 12 % of samples corrupted
y_outlier    = y_clean.copy()
n_outliers   = int(N * outlier_frac)
out_idx      = rng.choice(N, n_outliers, replace=False)
# Large additive deviations: uniform in [-30, -15] ∪ [15, 30]
sign    = rng.choice([-1, 1], n_outliers)
deviations = sign * rng.uniform(15, 30, n_outliers)
y_outlier[out_idx] += deviations

print(f"\nA3 Outlier Dataset:")
print(f"  Based on A1 clean data; {outlier_frac*100:.0f}% of labels corrupted")
print(f"  Corruption: additive uniform ±[15, 30]  →  {n_outliers} samples affected")


# ══════════════════════════════════════════════════════════════════
# PART B — FROM-SCRATCH IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART B — Algorithm Implementations")
print("=" * 60)

# ── B1: Closed-form OLS  w = (XᵀX)⁻¹ Xᵀy ────────────────────────
def ols_closed_form(X, y):
    """Closed-form OLS: w = (XᵀX)⁻¹ Xᵀy"""
    return np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]


# ── B2: Gradient Descent OLS ─────────────────────────────────────
def ols_gradient_descent(X, y, lr=1e-5, n_iter=2000):
    """Gradient descent on MSE: ∇w L = (2/N) Xᵀ(Xw - y)"""
    N, d = X.shape
    w    = np.zeros(d)
    losses = []
    for _ in range(n_iter):
        residual = X @ w - y
        grad     = (2 / N) * (X.T @ residual)
        w        = w - lr * grad
        losses.append(np.mean(residual ** 2))
    return w, losses


# ── B3: Ridge Regression ─────────────────────────────────────────
def ridge_closed_form(X, y, lam):
    """Ridge: w = (XᵀX + λI)⁻¹ Xᵀy"""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_gradient_descent(X, y, lam, lr=1e-5, n_iter=2000):
    """Ridge GD: ∇w L = (2/N) Xᵀ(Xw - y) + 2λw"""
    N, d = X.shape
    w    = np.zeros(d)
    losses = []
    for _ in range(n_iter):
        residual = X @ w - y
        grad     = (2 / N) * (X.T @ residual) + 2 * lam * w
        w        = w - lr * grad
        losses.append(np.mean(residual ** 2))
    return w, losses


# ── B4: Lasso via Sub-gradient Descent ───────────────────────────
def lasso_subgradient(X, y, lam, lr=1e-5, n_iter=3000):
    """
    Lasso sub-gradient: ∇w L = (2/N) Xᵀ(Xw - y) + λ·sign(w)
    sign(0) = 0 by convention.
    """
    N, d = X.shape
    w    = np.zeros(d)
    losses = []
    for _ in range(n_iter):
        residual = X @ w - y
        grad     = (2 / N) * (X.T @ residual) + lam * np.sign(w)
        w        = w - lr * grad
        losses.append(np.mean(residual ** 2))
    return w, losses


# ── B5: Weighted Least Squares  w = (XᵀΓX)⁻¹ XᵀΓy ──────────────
def weighted_least_squares(X, y, gamma):
    """Closed-form WLS: Γ = diag(γ), w = (XᵀΓX)⁻¹ XᵀΓy"""
    Gamma = np.diag(gamma)
    return np.linalg.solve(X.T @ Gamma @ X, X.T @ Gamma @ y)


def mse(X, y, w):
    return np.mean((X @ w - y) ** 2)


print("B1–B5: All algorithms defined ✓")


# ══════════════════════════════════════════════════════════════════
# PART C — GRADIENT DESCENT ANALYSIS (clean dataset)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART C — Gradient Descent Analysis (Clean Dataset)")
print("=" * 60)

# Eigenvalues of XᵀX to determine convergence threshold
eigvals_clean = np.linalg.eigvalsh(X_clean.T @ X_clean)
lambda_max    = eigvals_clean.max()
converge_bound = 2 / lambda_max
print(f"\nEigenvalues of XᵀX : {np.sort(eigvals_clean)[::-1].round(2)}")
print(f"λ_max              : {lambda_max:.4f}")
print(f"Convergence bound  : η < 2/λ_max = {converge_bound:.6f}")

LR_CONVERGE  = converge_bound * 0.4   # well inside bound — converges
LR_OSCILLATE = converge_bound * 0.98  # near bound — oscillates
LR_DIVERGE   = converge_bound * 1.5   # beyond bound — diverges

print(f"\nLearning rates tested:")
print(f"  Converges   η = {LR_CONVERGE:.6f}")
print(f"  Oscillates  η = {LR_OSCILLATE:.6f}")
print(f"  Diverges    η = {LR_DIVERGE:.6f}")

_, losses_conv = ols_gradient_descent(X_clean, y_clean, lr=LR_CONVERGE,  n_iter=500)
_, losses_osc  = ols_gradient_descent(X_clean, y_clean, lr=LR_OSCILLATE, n_iter=500)

# For diverge, only run a few iterations before values explode
losses_div = []
w_div = np.zeros(X_clean.shape[1])
for i in range(100):
    residual = X_clean @ w_div - y_clean
    grad     = (2 / N) * (X_clean.T @ residual)
    w_div    = w_div - LR_DIVERGE * grad
    loss     = np.mean(residual ** 2)
    losses_div.append(loss)
    if loss > 1e12 or np.isnan(loss):
        losses_div += [np.nan] * (100 - i - 1)
        break

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("C1: Loss vs. Iteration for Three Learning Rates", fontweight="bold")

configs = [
    (losses_conv, LR_CONVERGE,  "seagreen", "Converges"),
    (losses_osc,  LR_OSCILLATE, "darkorange", "Oscillates"),
    (losses_div,  LR_DIVERGE,   "tomato", "Diverges"),
]
for ax, (losses, lr, color, label) in zip(axes, configs):
    valid = [(i, l) for i, l in enumerate(losses) if not np.isnan(l) and l < 1e10]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, color=color, lw=2)
    ax.set_title(f"{label}\nη = {lr:.2e}", fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.text(0.95, 0.95, f"η < 2/λ_max = {converge_bound:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig("Q3_C1_learning_rates.png", dpi=130)
plt.show()
print("Saved: Q3_C1_learning_rates.png")

print(f"\nC2 Verification: η={LR_CONVERGE:.2e} < 2/λ_max={converge_bound:.2e} → converges ✓")
print(f"                 η={LR_DIVERGE:.2e} > 2/λ_max={converge_bound:.2e} → diverges  ✓")


# ══════════════════════════════════════════════════════════════════
# PART D — ILL-CONDITIONING AND RIDGE (correlated dataset)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART D — Ill-Conditioning and Ridge Regression")
print("=" * 60)

# 80/20 split for correlated dataset
split = int(0.8 * N)
X_corr_tr, X_corr_te = X_corr[:split], X_corr[split:]
y_corr_tr, y_corr_te = y_corr[:split], y_corr[split:]

LAMBDAS_RIDGE = [0.001, 0.01, 0.1, 1, 10]
print(f"\nD1: Eigenvalues of XᵀX = {np.sort(eigvals_corr)[::-1].round(2)}")
print(f"    Condition number   = {cond_corr:.4e}")

weight_mags = []
test_mses   = []

for lam in LAMBDAS_RIDGE:
    w   = ridge_closed_form(X_corr_tr, y_corr_tr, lam)
    mse_te = mse(X_corr_te, y_corr_te, w)
    weight_mags.append(np.abs(w[1:]))   # exclude intercept
    test_mses.append(mse_te)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("D2: Ridge Regression on Correlated Features", fontweight="bold")

# Weight magnitudes
ax = axes[0]
for i, feat in enumerate(["x1", "x2", "x3"]):
    ax.semilogx(LAMBDAS_RIDGE, [w[i] for w in weight_mags], "o-", lw=2, label=feat)
ax.set(xlabel="λ", ylabel="|weight|", title="Weight Magnitudes vs. λ")
ax.legend()
ax.grid(alpha=0.3)

# Test MSE
axes[1].semilogx(LAMBDAS_RIDGE, test_mses, "s-", color="tomato", lw=2, ms=8)
axes[1].set(xlabel="λ", ylabel="Test MSE", title="Test MSE vs. λ")
axes[1].grid(alpha=0.3)

# Condition number improvement
print("\nD2: Condition number vs λ:")
for lam in LAMBDAS_RIDGE:
    M      = X_corr_tr.T @ X_corr_tr + lam * np.eye(X_corr_tr.shape[1])
    ev     = np.linalg.eigvalsh(M)
    kappa  = ev.max() / ev.min()
    print(f"  λ={lam:.3f}  → cond(XᵀX + λI) = {kappa:.2e}")

plt.tight_layout()
plt.savefig("Q3_D2_ridge_conditioning.png", dpi=130)
plt.show()
print("Saved: Q3_D2_ridge_conditioning.png")


# ══════════════════════════════════════════════════════════════════
# PART E — OUTLIER STRESS TEST
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART E — Outlier Stress Test")
print("=" * 60)

split = int(0.8 * N)
X_out_tr, X_out_te = X_clean[:split], X_clean[split:]
y_out_tr, y_out_te = y_outlier[:split], y_outlier[split:]

w_ols_cf, _  = ols_closed_form(X_out_tr, y_out_tr), None
w_ols_cf     = ols_closed_form(X_out_tr, y_out_tr)
w_ols_gd, _  = ols_gradient_descent(X_out_tr, y_out_tr, lr=1e-5, n_iter=3000)
w_ridge      = ridge_closed_form(X_out_tr, y_out_tr, lam=1.0)
w_lasso, _   = lasso_subgradient(X_out_tr, y_out_tr, lam=0.1, lr=1e-5, n_iter=3000)
gamma_ones   = np.ones(len(X_out_tr))
w_wls        = weighted_least_squares(X_out_tr, y_out_tr, gamma_ones)

models = [
    ("OLS Closed-Form", w_ols_cf, "steelblue"),
    ("OLS Grad Descent", w_ols_gd, "darkorange"),
    ("Ridge (λ=1)",      w_ridge,  "seagreen"),
    ("Lasso (λ=0.1)",    w_lasso,  "purple"),
]

print("\nE1: Training MSE and learned weights (w0=intercept, w1=slope):")
print(f"  {'Method':<22} {'Train MSE':>10}  {'w0':>8}  {'w1':>8}")
for name, w, _ in models:
    print(f"  {name:<22} {mse(X_out_tr, y_out_tr, w):>10.4f}  {w[0]:>8.4f}  {w[1]:>8.4f}")

# Scatter + regression lines
x_range = np.array([-5, 5])
X_range = np.column_stack([np.ones(2), x_range])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_clean[:split], y_outlier[:split], s=2, alpha=0.15, color="grey", label="Data")
outlier_mask = np.isin(np.arange(split), out_idx[out_idx < split])
ax.scatter(x_clean[:split][outlier_mask], y_outlier[:split][outlier_mask],
           s=10, color="red", alpha=0.4, label="Outliers")

# True line
ax.plot(x_range, a_true * x_range + b_true, "k--", lw=2, label="True Line")

for name, w, color in models:
    y_line = X_range @ w
    ax.plot(x_range, y_line, color=color, lw=2, label=name)

ax.set(xlabel="x", ylabel="y", title="E1: Regression Lines on Outlier Dataset")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("Q3_E1_outlier_regression.png", dpi=130)
plt.show()
print("Saved: Q3_E1_outlier_regression.png")


# ══════════════════════════════════════════════════════════════════
# PART F — ITERATIVELY REWEIGHTED LEAST SQUARES
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PART F — Iteratively Reweighted Least Squares")
print("=" * 60)

MAX_ITER   = 50
TOL        = 1e-6
PLOT_ITERS = [1, 5, 10, MAX_ITER]   # iterations to visualise

X_irls = X_out_tr
y_irls = y_out_tr
n_irls = len(y_irls)

gamma        = np.ones(n_irls)
w_irls_hist  = []        # weights w at each iteration
gam_hist     = []        # gamma distributions
w_prev       = np.zeros(X_irls.shape[1])

for it in range(1, MAX_ITER + 1):
    w_new = weighted_least_squares(X_irls, y_irls, gamma)
    residuals = y_irls - X_irls @ w_new
    gamma_new = 1.0 / (1.0 + np.abs(residuals))

    w_irls_hist.append(w_new.copy())
    gam_hist.append(gamma_new.copy())

    change = np.linalg.norm(w_new - w_prev)
    if change < TOL and it > 5:
        print(f"  Converged at iteration {it}  (||Δw|| = {change:.2e})")
        # pad history to MAX_ITER
        while len(w_irls_hist) < MAX_ITER:
            w_irls_hist.append(w_new.copy())
            gam_hist.append(gamma_new.copy())
        break
    w_prev = w_new
    gamma  = gamma_new
else:
    print(f"  Reached max iterations ({MAX_ITER})")

# ── F2: Evolution of regression line + weight distribution ───────
fig, axes = plt.subplots(2, len(PLOT_ITERS), figsize=(16, 8))
fig.suptitle("F2: IRLS — Regression Line and Weight Distribution Across Iterations",
             fontweight="bold")

for col, it in enumerate(PLOT_ITERS):
    it_idx  = min(it, len(w_irls_hist)) - 1
    w_it    = w_irls_hist[it_idx]
    gam_it  = gam_hist[it_idx]

    # Top row: scatter + line
    ax_top = axes[0, col]
    ax_top.scatter(x_clean[:n_irls], y_irls, s=2, alpha=0.1, color="grey")
    ax_top.scatter(x_clean[:n_irls][outlier_mask[:n_irls]],
                   y_irls[outlier_mask[:n_irls]],
                   s=10, color="red", alpha=0.4)
    y_line_irls  = X_range @ w_it
    y_line_ols   = X_range @ w_ols_cf
    y_line_ridge = X_range @ w_ridge
    ax_top.plot(x_range, y_line_irls,  color="purple",   lw=2, label="IRLS")
    ax_top.plot(x_range, y_line_ols,   color="steelblue", lw=1.5, ls="--", label="OLS")
    ax_top.plot(x_range, y_line_ridge, color="seagreen",  lw=1.5, ls=":", label="Ridge")
    ax_top.plot(x_range, a_true * x_range + b_true, "k--", lw=1.5, label="True")
    ax_top.set_title(f"Iteration {it}", fontsize=10)
    ax_top.set_xlabel("x"); ax_top.set_ylabel("y")
    if col == 0:
        ax_top.legend(fontsize=7)

    # Bottom row: weight (gamma) distribution
    ax_bot = axes[1, col]
    ax_bot.hist(gam_it, bins=40, color="purple", alpha=0.7, edgecolor="white", lw=0.3)
    ax_bot.axvline(gam_it[outlier_mask[:n_irls]].mean(), color="red",
                   ls="--", lw=1.5, label="Mean γ (outliers)")
    ax_bot.axvline(gam_it[~outlier_mask[:n_irls]].mean(), color="seagreen",
                   ls="--", lw=1.5, label="Mean γ (inliers)")
    ax_bot.set_xlabel("γ (weight)"); ax_bot.set_ylabel("Count")
    if col == 0:
        ax_bot.legend(fontsize=7)

plt.tight_layout()
plt.savefig("Q3_F2_irls_evolution.png", dpi=130)
plt.show()
print("Saved: Q3_F2_irls_evolution.png")

# ── F3 Summary ────────────────────────────────────────────────────
w_irls_final = w_irls_hist[-1]
print(f"\nF3 Final Results:")
print(f"  IRLS  final  w = {w_irls_final.round(4)}  MSE = {mse(X_out_te, y_out_te, w_irls_final):.4f}")
print(f"  OLS   (CF)   w = {w_ols_cf.round(4)}   MSE = {mse(X_out_te, y_out_te, w_ols_cf):.4f}")
print(f"  Ridge (λ=1)  w = {w_ridge.round(4)}    MSE = {mse(X_out_te, y_out_te, w_ridge):.4f}")
print(f"  True weights   = [b={b_true}, a={a_true}]")

mean_gamma_out = gam_hist[-1][outlier_mask[:n_irls]].mean()
mean_gamma_in  = gam_hist[-1][~outlier_mask[:n_irls]].mean()
print(f"\n  Mean γ (outliers) = {mean_gamma_out:.4f}")
print(f"  Mean γ (inliers)  = {mean_gamma_in:.4f}")
print(f"  → Outliers are down-weighted by factor ≈ {mean_gamma_in/mean_gamma_out:.1f}×")

print("\nQ3 Complete ✓")
print("\nGenerated files:")
for fname in ["Q3_C1_learning_rates.png", "Q3_D2_ridge_conditioning.png",
              "Q3_E1_outlier_regression.png", "Q3_F2_irls_evolution.png"]:
    print(f"  {fname}")