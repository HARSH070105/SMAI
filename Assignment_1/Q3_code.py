import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

FIG_DIR = "Q3_Ouput"
os.makedirs(FIG_DIR, exist_ok=True)

# Make plots larger and legible for reports
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def save_fig(name):
    fname = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"Saved: {fname}")

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def add_bias(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.c_[np.ones(X.shape[0]), X]

def generate_clean_dataset(N=10000, a=3.5, b=2.0, sigma=2.0):
    x = np.random.uniform(-10, 10, N)
    eps = np.random.normal(0, sigma, N)
    y = a * x + b + eps
    X = add_bias(x)
    return X, y, x

def generate_correlated_dataset(N=10000, corr=0.95):
    x1 = np.random.normal(0, 1, N)
    x2 = corr * x1 + np.random.normal(0, 0.05, N)
    X_raw = np.vstack([x1, x2]).T
    y = 4 * x1 + 2 * x2 + np.random.normal(0, 1, N)
    X = add_bias(X_raw)
    return X, y, X_raw

def generate_outlier_dataset(N=10000, outlier_frac=0.1, outlier_shift=50.0):
    X, y, x = generate_clean_dataset(N=N)
    idx = np.random.choice(N, int(outlier_frac * N), replace=False)
    y[idx] += np.random.normal(outlier_shift, 10, len(idx))  # structured outliers
    return X, y, x, idx

def ols_closed(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def gradient_descent(X, y, lr=1e-3, iterations=500, tol=1e-12):
    w = np.zeros(X.shape[1])
    losses = []
    for i in range(iterations):
        pred = X @ w
        err = pred - y
        grad = (2 / len(y)) * X.T @ err
        w = w - lr * grad
        loss = mse(y, X @ w)
        losses.append(loss)
        if not np.isfinite(loss) or np.isnan(loss):
            # return early with indication of divergence
            return w, np.array(losses)
        if i > 1 and abs(losses[-1] - losses[-2]) < tol:
            break
    return w, np.array(losses)

def ridge_closed(X, y, lmbd):
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lmbd * I) @ X.T @ y

def ridge_gd(X, y, lmbd, lr=1e-3, iterations=500):
    w = np.zeros(X.shape[1])
    losses = []
    for i in range(iterations):
        pred = X @ w
        grad = (2 / len(y)) * X.T @ (pred - y) + 2 * lmbd * w
        w = w - lr * grad
        losses.append(mse(y, X @ w))
        if not np.isfinite(losses[-1]):
            return w, np.array(losses)
    return w, np.array(losses)

def lasso_subgradient(X, y, lmbd, lr=1e-3, iterations=1000):
    w = np.zeros(X.shape[1])
    for i in range(iterations):
        pred = X @ w
        grad = (2 / len(y)) * X.T @ (pred - y)
        subgrad = lmbd * np.sign(w)
        w = w - lr * (grad + subgrad)
        if not np.all(np.isfinite(w)):
            break
    return w

def weighted_ls(X, y, gamma):
    # gamma: array length N
    G = np.diag(gamma)
    return np.linalg.inv(X.T @ G @ X) @ X.T @ G @ y

def irls_iter(X, y, max_iter=50, tol=1e-6):
    N = len(y)
    gamma = np.ones(N)
    history_w = []
    history_gamma = []
    for it in range(max_iter):
        w = weighted_ls(X, y, gamma)
        residuals = np.abs(y - X @ w)
        # guard numerical issues
        gamma_new = 1.0 / (1.0 + residuals)
        history_w.append(w)
        history_gamma.append(gamma_new.copy())
        if np.linalg.norm(gamma_new - gamma) < tol:
            break
        gamma = gamma_new
    return np.array(history_w), np.array(history_gamma)

def plot_gd_analysis():
    X, y, _ = generate_clean_dataset()
    XT_X = X.T @ X
    eigvals = np.linalg.eigvals(XT_X)
    lam_max = np.max(eigvals)
    print("Eigenvalues (clean X^T X):", eigvals)
    print("Max eigenvalue:", lam_max)
    print("GD convergence bound: lr < {:.3e}".format(2.0 / lam_max))

    lr_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-1]
    stable = {}
    unstable = {}

    for lr in lr_values:
        w, losses = gradient_descent(X, y, lr=lr, iterations=1000)
        if np.any(~np.isfinite(losses)) or np.nanmax(losses) > 1e50:
            unstable[lr] = losses
        else:
            stable[lr] = losses

    # stable (plot on semilogy to separate magnitudes)
    plt.figure()
    for lr, losses in stable.items():
        plt.semilogy(np.arange(len(losses)), losses, label=f"lr={lr}")
    plt.title("Gradient Descent — Convergent learning rates (log scale)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    save_fig("gd_convergent_semilogy.png")
    plt.show()

    # unstable (plot with clipped y for visibility)
    plt.figure()
    for lr, losses in unstable.items():
        # losses may contain inf — plot until it blows up
        finite_idx = np.where(np.isfinite(losses))[0]
        if finite_idx.size == 0:
            continue
        last = finite_idx[-1]
        plt.semilogy(np.arange(last + 1), losses[: last + 1], label=f"lr={lr}")
    plt.title("Gradient Descent — Divergent learning rates (log scale)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    save_fig("gd_divergent_semilogy.png")
    plt.show()

def plot_ridge_analysis():
    X, y, _ = generate_correlated_dataset()
    XT_X = X.T @ X
    eigvals = np.linalg.eigvals(XT_X)
    cond_before = np.linalg.cond(XT_X)
    print("Eigenvalues (corr X^T X):", eigvals)
    print("Condition number before ridge:", cond_before)

    lambdas = np.logspace(-4, 2, 40)
    weight_norms = []
    weights = []
    conds_after = []
    errors = []

    for l in lambdas:
        w = ridge_closed(X, y, l)
        weight_norms.append(np.linalg.norm(w))
        weights.append(w)
        conds_after.append(np.linalg.cond(XT_X + l * np.eye(X.shape[1])))
        errors.append(mse(y, X @ w))

    weights = np.array(weights)

    # Weight norms vs lambda
    plt.figure()
    plt.semilogx(lambdas, weight_norms, marker="o")
    plt.title("Ridge: ||w|| vs λ (log x)")
    plt.xlabel("λ")
    plt.ylabel("||w||")
    plt.grid(True, which="both", ls="--", lw=0.5)
    save_fig("ridge_weightnorm_vs_lambda.png")
    plt.show()

    # Individual coefficient traces (including bias)
    plt.figure()
    for j in range(weights.shape[1]):  # one curve per coefficient
        plt.semilogx(lambdas, weights[:, j], label=f"w[{j}]")
    plt.title("Ridge: Coefficient traces vs λ")
    plt.xlabel("λ")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    save_fig("ridge_coeff_traces.png")
    plt.show()

    # Condition number improvement
    plt.figure()
    plt.semilogx(lambdas, conds_after, marker="o")
    plt.title("Condition number of (X^T X + λI) vs λ")
    plt.xlabel("λ")
    plt.ylabel("Condition number")
    plt.grid(True, which="both", ls="--", lw=0.5)
    save_fig("ridge_condition_vs_lambda.png")
    plt.show()

    # Error vs lambda
    plt.figure()
    plt.semilogx(lambdas, errors, marker="o")
    plt.title("Training MSE vs λ (Ridge)")
    plt.xlabel("λ")
    plt.ylabel("MSE")
    plt.grid(True, which="both", ls="--", lw=0.5)
    save_fig("ridge_mse_vs_lambda.png")
    plt.show()

def plot_outlier_experiment():
    X, y, x, out_idx = generate_outlier_dataset()
    w_ols = ols_closed(X, y)
    w_ridge = ridge_closed(X, y, 1.0)
    w_lasso = lasso_subgradient(X, y, lmbd=0.1, lr=1e-3, iterations=2000)

    print("Training MSE (Outlier dataset):")
    print("OLS:", mse(y, X @ w_ols))
    print("Ridge:", mse(y, X @ w_ridge))
    print("Lasso:", mse(y, X @ w_lasso))

    xs = np.linspace(np.min(x), np.max(x), 300)
    Xp = add_bias(xs)

    # Full scale plot (shows outliers clearly)
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=5, alpha=0.4, label="data")
    plt.plot(xs, Xp @ w_ols, label="OLS", linewidth=2)
    plt.plot(xs, Xp @ w_ridge, label="Ridge (λ=1)", linewidth=2)
    plt.plot(xs, Xp @ w_lasso, label="Lasso (λ=0.1)", linewidth=2)
    plt.title("Regression lines on Outlier dataset — Full scale")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    save_fig("outliers_full_scale.png")
    plt.show()

    # Zoomed-in view on main cloud (separate plot) to compare fitted lines for majority
    plt.figure(figsize=(9, 6))
    # choose y-limits to focus on main cloud (exclude high outliers)
    y_low, y_high = np.percentile(y, [1, 90])
    plt.ylim([y_low - 5, y_high + 5])
    plt.scatter(x, y, s=5, alpha=0.4, label="data (zoom)")
    plt.plot(xs, Xp @ w_ols, label="OLS", linewidth=2)
    plt.plot(xs, Xp @ w_ridge, label="Ridge (λ=1)", linewidth=2)
    plt.plot(xs, Xp @ w_lasso, label="Lasso (λ=0.1)", linewidth=2)
    plt.title("Regression lines on Outlier dataset — Zoomed main cloud")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    save_fig("outliers_zoomed.png")
    plt.show()

    # show which points were flagged as outliers (indices)
    print(f"Number of injected outliers: {len(out_idx)}")
    # small scatter showing outliers highlighted
    plt.figure(figsize=(9, 4))
    plt.scatter(x, y, s=4, alpha=0.25)
    plt.scatter(x[out_idx], y[out_idx], s=10, color="red", alpha=0.8, label="injected outliers")
    plt.title("Injected outliers (red)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    save_fig("outliers_highlighted.png")
    plt.show()

def plot_irls_demo():
    X, y, x, out_idx = generate_outlier_dataset()
    history_w, history_gamma = irls_iter(X, y, max_iter=50)

    # Plot regression lines at selected iterations
    xs = np.linspace(np.min(x), np.max(x), 300)
    Xp = add_bias(xs)

    iters_to_plot = [0, 1, 4, min(9, len(history_w) - 1), len(history_w) - 1]
    # unique and sorted
    iters_to_plot = sorted(set([i for i in iters_to_plot if i >= 0 and i < len(history_w)]))

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=5, alpha=0.3)
    for i in iters_to_plot:
        w = history_w[i]
        plt.plot(xs, Xp @ w, label=f"iter {i+1}", linewidth=2)
    plt.title("IRLS Regression line evolution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    save_fig("irls_line_evolution.png")
    plt.show()

    # Plot gamma summaries: histogram at start, middle, final
    plt.figure(figsize=(9, 4))
    subplot_positions = [0, len(history_gamma) // 2, len(history_gamma) - 1]
    for i, pos in enumerate(subplot_positions):
        plt.subplot(1, 3, i + 1)
        plt.hist(history_gamma[pos], bins=30)
        plt.title(f"gamma distribution\niter {pos+1}")
        plt.xlabel("gamma")
    save_fig("irls_gamma_histograms.png")
    plt.show()

    # Plot mean gamma vs iteration (shows down-weighting trend)
    mean_gamma = np.mean(history_gamma, axis=1)
    plt.figure()
    plt.plot(np.arange(1, len(mean_gamma) + 1), mean_gamma, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("mean(gamma)")
    plt.title("IRLS: mean gamma vs iteration (down-weighting outliers)")
    plt.grid(True, ls="--", lw=0.5)
    save_fig("irls_mean_gamma_vs_iter.png")
    plt.show()

def main():
    print("\nRunning GD analysis and saving figures...")
    plot_gd_analysis()

    print("\nRunning ridge analysis and saving figures...")
    plot_ridge_analysis()

    print("\nRunning outlier experiment and saving figures...")
    plot_outlier_experiment()

    print("\nRunning IRLS demo and saving figures...")
    plot_irls_demo()

    # Final summary numeric printouts for inclusion in report
    Xc, yc, _ = generate_clean_dataset()
    XT_X = Xc.T @ Xc
    eigvals_clean = np.linalg.eigvals(XT_X)
    print("\nSummary (clean dataset):")
    print("X^T X eigenvalues:", eigvals_clean)
    print("max eigenvalue:", np.max(eigvals_clean))
    print("GD convergence threshold lr < {:.3e}".format(2.0 / np.max(eigvals_clean)))

    Xcorr, ycorr, Xraw_corr = generate_correlated_dataset()
    XT_X_corr = Xcorr.T @ Xcorr
    eigvals_corr = np.linalg.eigvals(XT_X_corr)
    print("\nSummary (correlated dataset):")
    print("X^T X eigenvalues:", eigvals_corr)
    print("condition number:", np.linalg.cond(XT_X_corr))

    Xo, yo, xo, _ = generate_outlier_dataset()
    w_ols = ols_closed(Xo, yo)
    w_ridge = ridge_closed(Xo, yo, 1.0)
    w_lasso = lasso_subgradient(Xo, yo, 0.1, lr=1e-3, iterations=2000)
    print("\nFinal training MSEs (outlier dataset):")
    print("OLS:", mse(yo, Xo @ w_ols))
    print("Ridge:", mse(yo, Xo @ w_ridge))
    print("Lasso:", mse(yo, Xo @ w_lasso))

    print("\nAll figures saved in directory:", FIG_DIR)

if __name__ == "__main__":
    main()