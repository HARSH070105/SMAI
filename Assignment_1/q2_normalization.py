"""
SMAI Spring 2026 — Assignment 1
Question 2: Data Normalisation and KNN Classification
[CORRECTED & ENHANCED VERSION]
------------------------------------------------------
HOW TO RUN:
  1. Set ROLL_NUMBER below to your university roll number.
  2. Set API_URL to the actual endpoint (or leave it to use fallback synthetic data).
  3. Run:  python Q2_normalisation_knn_FIXED.py
  Requirements: numpy, pandas, matplotlib, seaborn, requests, scipy
  Install:      pip install numpy pandas matplotlib seaborn requests scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import math
from collections import Counter
from scipy.special import erfinv

warnings.filterwarnings("ignore")
np.random.seed(42)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURE THESE BEFORE RUNNING                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
ROLL_NUMBER = 2023112004          # <-- Replace with your actual roll number
API_URL     = "http://preon.iiit.ac.in:8026/api/data"  # Base URL (params added dynamically)
N_SAMPLES   = 1500       # Collect this many samples (>= 1000 required)

SCENARIO_ID  = ROLL_NUMBER % 8
SCENARIO_MAP = {
    0: "The Skewed Reality      — Log-Normal / Power-Law features, long-tail outliers",
    1: "The Broken Sensor       — Gaussian + massive random noise (±10,000)",
    2: "Apples & Oranges        — Vastly different feature scales (e.g. Age vs Salary)",
    3: "The Direction Vector    — High-dim data; angle matters more than magnitude",
    4: "The Double Peak         — Bimodal distribution; single mean falls in valley",
    5: "The Hard Limit          — Data bunched against a maximum/minimum (censored)",
    6: "The Negative Skew       — Long left tail; negative values present",
    7: "The Sparse Manhattan    — Integer counts with many zeros; L1 preferred",
}
print(f"Roll Number  : {ROLL_NUMBER}")
print(f"Scenario ID  : {SCENARIO_ID}")
print(f"Scenario     : {SCENARIO_MAP[SCENARIO_ID]}")
print()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════

def fetch_dataset(roll_number: int, n_samples: int, api_url: str) -> pd.DataFrame:
    """Poll the API index-by-index and collect samples into a DataFrame."""
    records = []
    print(f"Fetching {n_samples} samples from API …")
    for idx in range(n_samples):
        try:
            # FIX: Use correct parameter names (roll, index) and proper params dict
            resp = requests.get(api_url, params={"roll": roll_number, "index": idx}, timeout=10)
            data = resp.json()
            if data is None:
                print(f"  Reached end of dataset at index {idx}.")
                break
            row = data["features"] + [data["label"]]
            records.append(row)
        except Exception as e:
            print(f"  Warning: index {idx} failed — {e}")
        if (idx + 1) % 200 == 0:
            print(f"  … collected {idx+1} samples")

    n_features = len(records[0]) - 1
    cols = [f"feature_{i}" for i in range(n_features)] + ["label"]
    df   = pd.DataFrame(records, columns=cols)
    print(f"Collection complete: {len(df)} rows, {n_features} features.\n")
    return df


# ── Try fetching; fall back to synthetic data if API is not configured ──────
try:
    # Quick test to see if API is reachable
    resp = requests.head(API_URL, timeout=5)
    df_raw = fetch_dataset(ROLL_NUMBER, N_SAMPLES, API_URL)
except Exception as e:
    print(f"⚠  API not reachable ({e}) — generating SYNTHETIC data for demonstration.")
    print("   Set ROLL_NUMBER and API_URL at the top of this file for real data.\n")
    rng = np.random.RandomState(ROLL_NUMBER if ROLL_NUMBER else 42)
    if SCENARIO_ID == 0:                         # Skewed
        X_raw = rng.lognormal(0, 1.5, (N_SAMPLES, 4))
    elif SCENARIO_ID == 1:                       # Broken sensor
        X_raw = rng.randn(N_SAMPLES, 4)
        mask  = rng.rand(N_SAMPLES, 4) < 0.05
        X_raw[mask] += rng.choice([-10000, 10000], mask.sum())
    elif SCENARIO_ID == 2:                       # Apples & oranges
        X_raw = np.column_stack([
            rng.randint(18, 80, N_SAMPLES).astype(float),
            rng.normal(50000, 15000, N_SAMPLES),
            rng.normal(0, 1, N_SAMPLES),
            rng.normal(0, 1, N_SAMPLES),
        ])
    elif SCENARIO_ID == 3:                       # Direction vectors
        X_raw = rng.randn(N_SAMPLES, 10)
        X_raw *= rng.exponential(2, (N_SAMPLES, 1))
    elif SCENARIO_ID == 4:                       # Bimodal
        n1   = N_SAMPLES // 2
        X_raw = np.column_stack([
            np.concatenate([rng.normal(-3, 0.5, n1), rng.normal(3, 0.5, N_SAMPLES - n1)]),
            rng.randn(N_SAMPLES),
        ])
    elif SCENARIO_ID == 5:                       # Hard limit
        X_raw = np.clip(rng.exponential(0.5, (N_SAMPLES, 4)), 0, 1)
    elif SCENARIO_ID == 6:                       # Negative skew
        X_raw = -rng.lognormal(0, 1, (N_SAMPLES, 4))
    else:                                        # Sparse Manhattan
        X_raw = rng.poisson(0.5, (N_SAMPLES, 8)).astype(float)

    # Simple synthetic labels
    weights = rng.randn(X_raw.shape[1])
    logits  = X_raw @ weights
    labels  = (logits > np.median(logits)).astype(int)
    df_raw  = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])
    df_raw["label"] = labels

FEATURE_COLS = [c for c in df_raw.columns if c != "label"]
X_all = df_raw[FEATURE_COLS].values.astype(float)
y_all = df_raw["label"].values

print(f"Dataset shape : {X_all.shape}")
print(f"Classes       : {np.unique(y_all)}")


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    rng   = np.random.RandomState(random_state)
    idx   = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_train, X_test, y_train, y_test = train_test_split_manual(X_all, y_all, test_size=0.2, random_state=42)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}\n")


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — NORMALISATION METHODS (all 20 from scratch)
# ══════════════════════════════════════════════════════════════════

class Normaliser:
    """Base class — fit on training data, transform any split."""
    def fit(self, X): raise NotImplementedError
    def transform(self, X): raise NotImplementedError
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ── A. Linear Scaling ────────────────────────────────────────────
class MinMaxScaler(Normaliser):
    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
    def transform(self, X):
        denom = self.xmax - self.xmin
        denom = np.where(denom == 0, 1, denom)   # avoid divide-by-zero
        return (X - self.xmin) / denom


class MeanNormalisation(Normaliser):
    def fit(self, X):
        self.mu   = X.mean(axis=0)
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
    def transform(self, X):
        denom = self.xmax - self.xmin
        denom = np.where(denom == 0, 1, denom)
        return (X - self.mu) / denom


class MaxAbsoluteScaler(Normaliser):
    def fit(self, X):
        self.max_abs = np.abs(X).max(axis=0)
        self.max_abs = np.where(self.max_abs == 0, 1, self.max_abs)
    def transform(self, X):
        return X / self.max_abs


class DecimalScaling(Normaliser):
    def fit(self, X):
        max_abs = np.abs(X).max(axis=0)
        self.j  = np.ceil(np.log10(np.where(max_abs == 0, 1, max_abs))).astype(int)
    def transform(self, X):
        return X / (10 ** self.j)


class RobustScaler(Normaliser):
    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.q1     = np.percentile(X, 25, axis=0)
        self.q3     = np.percentile(X, 75, axis=0)
        iqr         = self.q3 - self.q1
        self.iqr    = np.where(iqr == 0, 1, iqr)
    def transform(self, X):
        return (X - self.median) / self.iqr


# ── B. Statistical Normalisation ────────────────────────────────
class ZScoreStandardiser(Normaliser):
    def fit(self, X):
        self.mu  = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
    def transform(self, X):
        return (X - self.mu) / self.std


class ModifiedZScore(Normaliser):
    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.mad    = np.median(np.abs(X - self.median), axis=0)
        self.mad    = np.where(self.mad == 0, 1, self.mad)
    def transform(self, X):
        return 0.6745 * (X - self.median) / self.mad


class ParetoScaling(Normaliser):
    def fit(self, X):
        self.mu  = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.sqrt_std = np.where(self.std == 0, 1, np.sqrt(self.std))
    def transform(self, X):
        return (X - self.mu) / self.sqrt_std


# ── C. Non-Linear Transformations ────────────────────────────────
class LogTransformation(Normaliser):
    """c is chosen so that X + c > 0 for all values."""
    def fit(self, X):
        self.c = np.where(X.min(axis=0) <= 0, np.abs(X.min(axis=0)) + 1, 0)
    def transform(self, X):
        return np.log(X + self.c)


class ReciprocalTransformation(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        safe = np.where(X == 0, np.finfo(float).eps, X)
        result = 1.0 / safe
        # FIX: Clip extreme values to prevent overflow
        return np.clip(result, -1e10, 1e10)


class SquareRootTransformation(Normaliser):
    """Shift negatives to >=0 before sqrt."""
    def fit(self, X):
        self.shift = np.where(X.min(axis=0) < 0, np.abs(X.min(axis=0)), 0)
    def transform(self, X):
        return np.sqrt(X + self.shift)


class BoxCoxLambdaHalf(Normaliser):
    """Box-Cox with lambda=0.5; requires X > 0, so shift if needed."""
    def fit(self, X):
        self.shift = np.where(X.min(axis=0) <= 0, np.abs(X.min(axis=0)) + 1, 0)
    def transform(self, X):
        Xp = X + self.shift
        lam = 0.5
        return (np.power(Xp, lam) - 1) / lam


class YeoJohnsonLambdaHalf(Normaliser):
    """Yeo-Johnson with lambda=0.5; handles any real values."""
    def fit(self, X): pass
    def transform(self, X):
        lam = 0.5
        result = np.zeros_like(X, dtype=float)
        pos    = X >= 0
        neg    = ~pos
        result[pos] = (np.power(X[pos] + 1, lam) - 1) / lam
        result[neg] = -(np.power(-X[neg] + 1, 2 - lam) - 1) / (2 - lam)
        return result


class HyperbolicTangent(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        return np.tanh(X)


class SigmoidLogistic(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        # FIX: Clip to prevent overflow in exp()
        X_clipped = np.clip(X, -500, 500)
        return 1.0 / (1.0 + np.exp(-X_clipped))


# ── D. Vector Normalisation (row-wise) ───────────────────────────
class L1Normalisation(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        norms = np.abs(X).sum(axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms


class L2Normalisation(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms


class SoftmaxScaling(Normaliser):
    """Row-wise softmax."""
    def fit(self, X): pass
    def transform(self, X):
        # Subtract row max for numerical stability
        Xs = X - X.max(axis=1, keepdims=True)
        ex = np.exp(Xs)
        return ex / ex.sum(axis=1, keepdims=True)


# ── E. Distribution Mapping ──────────────────────────────────────
class QuantileNormalisationUniform(Normaliser):
    """Maps each feature to a uniform [0,1] distribution using rank."""
    def fit(self, X):
        self.n_train = X.shape[0]
    def transform(self, X):
        n = X.shape[0]
        result = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = np.argsort(np.argsort(X[:, j]))
            result[:, j] = (ranks + 0.5) / n    # +0.5 avoids 0 and 1
        return result


class RankGauss(Normaliser):
    """Maps ranks to Gaussian via inverse CDF (erfinv)."""
    def fit(self, X):
        self.n_train = X.shape[0]
    def transform(self, X):
        n = X.shape[0]
        result = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = np.argsort(np.argsort(X[:, j]))
            u     = (ranks + 0.5) / n           # uniform (0,1)
            # Clamp to avoid ±inf from erfinv
            u     = np.clip(u, 1e-6, 1 - 1e-6)
            result[:, j] = np.sqrt(2) * erfinv(2 * u - 1)
        return result


# ── Method registry ──────────────────────────────────────────────
NORMALISERS = {
    "1. Min-Max Scaling":           MinMaxScaler(),
    "2. Mean Normalisation":        MeanNormalisation(),
    "3. Max Absolute Scaling":      MaxAbsoluteScaler(),
    "4. Decimal Scaling":           DecimalScaling(),
    "5. Robust Scaling (IQR)":      RobustScaler(),
    "6. Z-Score Standardisation":   ZScoreStandardiser(),
    "7. Modified Z-Score":          ModifiedZScore(),
    "8. Pareto Scaling":            ParetoScaling(),
    "9. Log Transformation":        LogTransformation(),
    "10. Reciprocal Transformation":ReciprocalTransformation(),
    "11. Square Root Transform":    SquareRootTransformation(),
    "12. Box-Cox (λ=0.5)":          BoxCoxLambdaHalf(),
    "13. Yeo-Johnson (λ=0.5)":      YeoJohnsonLambdaHalf(),
    "14. Hyperbolic Tangent":       HyperbolicTangent(),
    "15. Sigmoid / Logistic":       SigmoidLogistic(),
    "16. L1 Normalisation":         L1Normalisation(),
    "17. L2 Normalisation":         L2Normalisation(),
    "18. Softmax Scaling":          SoftmaxScaling(),
    "19. Quantile (Uniform)":       QuantileNormalisationUniform(),
    "20. Rank-Gauss":               RankGauss(),
}


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — KNN FROM SCRATCH
# ══════════════════════════════════════════════════════════════════

def knn_predict(X_train, y_train, X_test, k):
    """Euclidean-distance KNN prediction."""
    # Compute pairwise distances efficiently
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_train = (X_train ** 2).sum(axis=1)
    sq_test  = (X_test  ** 2).sum(axis=1)
    dists_sq = sq_test[:, None] + sq_train[None, :] - 2 * (X_test @ X_train.T)
    dists_sq = np.maximum(dists_sq, 0)             # numerical safety

    preds = []
    for row_dists in dists_sq:
        nn_idx = np.argpartition(row_dists, min(k, len(row_dists)-1))[:k]
        votes  = Counter(y_train[nn_idx])
        preds.append(votes.most_common(1)[0][0])
    return np.array(preds)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)


def kfold_cv_knn(X, y, k_values, n_folds=5, random_state=42):
    """5-fold CV to select best k. Returns dict {k: mean_cv_acc}."""
    rng   = np.random.RandomState(random_state)
    idx   = np.arange(len(X))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    cv_accs = {k: [] for k in k_values}
    for fold_i in range(n_folds):
        val_idx   = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
        Xtr, ytr  = X[train_idx], y[train_idx]
        Xvl, yvl  = X[val_idx],   y[val_idx]
        for k in k_values:
            preds = knn_predict(Xtr, ytr, Xvl, k)
            cv_accs[k].append(accuracy(yvl, preds))
    return {k: np.mean(v) for k, v in cv_accs.items()}


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════

K_VALUES = list(range(1, 31))
results  = []

# ── Baseline (raw data) ──────────────────────────────────────────
print("Evaluating Raw Data (baseline) …")
cv_raw  = kfold_cv_knn(X_train, y_train, K_VALUES)
best_k  = max(cv_raw, key=cv_raw.get)
y_pred  = knn_predict(X_train, y_train, X_test, best_k)
acc_raw = accuracy(y_test, y_pred)
f1_raw  = macro_f1(y_test, y_pred)
results.append({
    "Method":   "Raw Data (baseline)",
    "Best k":   best_k,
    "CV Accs":  cv_raw,
    "Test Acc": acc_raw,
    "Macro F1": f1_raw,
})
print(f"  Best k={best_k}  Test Acc={acc_raw:.4f}  F1={f1_raw:.4f}")

# ── 20 normalisation methods ─────────────────────────────────────
for name, norm in NORMALISERS.items():
    print(f"Evaluating {name} …")
    try:
        norm.fit(X_train)
        Xtr_n = norm.transform(X_train)
        Xte_n = norm.transform(X_test)
        # Replace any NaN/Inf with 0
        Xtr_n = np.nan_to_num(Xtr_n, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_n = np.nan_to_num(Xte_n, nan=0.0, posinf=0.0, neginf=0.0)

        cv_scores = kfold_cv_knn(Xtr_n, y_train, K_VALUES)
        best_k    = max(cv_scores, key=cv_scores.get)
        y_pred    = knn_predict(Xtr_n, y_train, Xte_n, best_k)
        acc       = accuracy(y_test, y_pred)
        f1        = macro_f1(y_test, y_pred)
        results.append({
            "Method":   name,
            "Best k":   best_k,
            "CV Accs":  cv_scores,
            "Test Acc": acc,
            "Macro F1": f1,
        })
        print(f"  Best k={best_k}  Test Acc={acc:.4f}  F1={f1:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"Method": name, "Best k": "N/A", "CV Accs": {}, "Test Acc": 0.0, "Macro F1": 0.0})


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS TABLE
# ══════════════════════════════════════════════════════════════════

df_results = pd.DataFrame([{
    "Method":    r["Method"],
    "Best k":    r["Best k"],
    "Test Acc":  round(r["Test Acc"], 4),
    "Macro F1":  round(r["Macro F1"], 4),
} for r in results])

df_results["Rank"] = df_results["Test Acc"].rank(ascending=False, method="min").astype(int)
df_results = df_results.sort_values("Rank").reset_index(drop=True)

print("\n" + "═" * 72)
print("  MASTER RESULTS TABLE")
print("═" * 72)
print(df_results.to_string(index=False))
print()

best_method  = df_results.iloc[0]["Method"]
worst_method = df_results.iloc[-1]["Method"]
print(f"Best  method : {best_method}")
print(f"Worst method : {worst_method}")


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — VISUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════

FEAT_IDX = 0   # Feature to inspect for distribution plots

# ── Helper: get transformed feature values ───────────────────────
def get_transformed_col(method_name, col_idx):
    if method_name == "Raw Data (baseline)":
        return X_all[:, col_idx]
    norm = NORMALISERS.get(method_name)
    if norm is None:
        return X_all[:, col_idx]
    norm.fit(X_train)
    Xtr_n = norm.transform(X_train)
    Xte_n = norm.transform(X_test)
    combined = np.concatenate([Xtr_n, Xte_n], axis=0)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    return combined[:, col_idx]


# ── V.1 Feature Distribution Plot ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Feature {FEAT_IDX} Distribution: Raw vs Best vs Worst", fontweight="bold")

configs = [
    ("Raw Data (baseline)", "steelblue"),
    (best_method, "seagreen"),
    (worst_method, "tomato"),
]

for ax, (mname, color) in zip(axes, configs):
    vals = get_transformed_col(mname, FEAT_IDX)
    # Clip extreme values for plotting
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    vals_clipped = np.clip(vals, lo, hi)
    ax.hist(vals_clipped, bins=40, color=color, edgecolor="white", lw=0.4, alpha=0.85)
    ax.set_title(mname[:40], fontsize=9)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Q2_V1_feature_distribution.png", dpi=130, bbox_inches='tight')
plt.show()
print("Saved: Q2_V1_feature_distribution.png")


# ── V.2 Accuracy vs k Plot ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Cross-Validation Accuracy vs. k — Raw / Best / Worst Method", fontweight="bold")

for r in results:
    if r["Method"] in ["Raw Data (baseline)", best_method, worst_method]:
        cv_accs = r["CV Accs"]
        if not cv_accs:
            continue
        ks   = sorted(cv_accs.keys())
        accs = [cv_accs[k] for k in ks]
        style = {"Raw Data (baseline)": ("steelblue", "--"),
                 best_method:            ("seagreen", "-"),
                 worst_method:           ("tomato", ":")}.get(r["Method"], ("grey", "-"))
        ax.plot(ks, accs, color=style[0], ls=style[1], lw=2, label=r["Method"][:35])

ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("CV Accuracy")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/Q2_V2_accuracy_vs_k.png", dpi=130, bbox_inches='tight')
plt.show()
print("Saved: Q2_V2_accuracy_vs_k.png")

# ── Save results table to CSV ─────────────────────────────────────
df_results.to_csv("/mnt/user-data/outputs/Q2_results_table.csv", index=False)
print("Saved: Q2_results_table.csv")
print("\n✓ Q2 Complete!")