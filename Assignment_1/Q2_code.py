import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import os
import sys
import pickle
from collections import Counter
from scipy.special import erfinv

warnings.filterwarnings("ignore")
np.random.seed(42)


ROLL_NUMBER = 2023112004
SCENARIO_ID = 4
N_SAMPLES = 2000
API_URL = "http://preon.iiit.ac.in:8026/api/data"
API_TIMEOUT = 10

SCENARIO_MAP = {
    4: "The Double Peak — Bimodal distribution; single mean falls in valley"
}

OUTPUT_DIR = "Q2_Output"
DATA_FILE = os.path.join(OUTPUT_DIR, "dataset.pkl")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "visualizations")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "Q2_results_table.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print("=" * 80)
print("SMAI — Question 2: Data Normalisation & KNN Classification")
print("=" * 80)
print(f"Roll Number  : {ROLL_NUMBER}")
print(f"Scenario ID  : {SCENARIO_ID}")
print(f"Scenario     : {SCENARIO_MAP[SCENARIO_ID]}")
print()

print("=" * 80)
print("STEP 1: DATA COLLECTION & STORAGE")
print("=" * 80)
print()

def save_dataset(X_all, y_all, source):
    """Save dataset locally."""
    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'X': X_all, 'y': y_all, 'source': source}, f)
    print(f"Dataset saved to: {os.path.abspath(DATA_FILE)}")

def load_dataset():
    """Load dataset from storage."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y'], data['source']
    return None, None, None

def fetch_from_api(roll_number, n_samples, api_url, timeout):
    """Fetch data from API."""
    records = []
    print(f"Fetching {n_samples} samples from API: {api_url}")
    print(f"Parameters: roll={roll_number}, indices=0 to {n_samples-1}")
    print()
    
    for idx in range(n_samples):
        try:
            resp = requests.get(
                api_url, 
                params={"roll": roll_number, "index": idx}, 
                timeout=timeout
            )
            data = resp.json()
            if data is None:
                print(f"  Reached end of dataset at index {idx}.")
                break
            row = data["features"] + [data["label"]]
            records.append(row)
        except Exception as e:
            if idx < 3:
                print(f"  Index {idx}: {type(e).__name__}")
        
        
        bar_len = 80
        if (idx + 1) % 10 == 0 or idx == n_samples - 1:
            progress = (idx + 1) / n_samples
            filled_len = int(bar_len * progress)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r  [{bar}] {idx+1}/{n_samples} samples", end='', flush=True)
        if idx == n_samples - 1:
            print()  # Newline after last update
    
    if not records:
        return None
    
    n_features = len(records[0]) - 1
    cols = [f"feature_{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(records, columns=cols)
    return df

# Check if dataset exists locally
print("Checking for stored dataset…")
X_all, y_all, data_source = load_dataset()

if X_all is not None:
    print(f"  Loaded dataset from storage (source: {data_source})")
    print(f"  Shape: {X_all.shape}")
    print()
else:
    print("  No stored dataset found. Attempting API fetch…")
    print()
    
    df_raw = fetch_from_api(ROLL_NUMBER, N_SAMPLES, API_URL, API_TIMEOUT)
    
    if df_raw is not None and len(df_raw) >= 1000:
        print(f"  Successfully fetched {len(df_raw)} samples from API")
        print()
        FEATURE_COLS = [c for c in df_raw.columns if c != "label"]
        X_all = df_raw[FEATURE_COLS].values.astype(float)
        y_all = df_raw["label"].values
        data_source = "API"
        save_dataset(X_all, y_all, data_source)


print(f"Data Source   : {data_source}")
print(f"Dataset shape : {X_all.shape}")
print(f"Classes       : {np.unique(y_all)}")
print(f"Class balance : {np.bincount(y_all)}")
print()


def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_train, X_test, y_train, y_test = train_test_split_manual(X_all, y_all, test_size=0.2, random_state=42)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print()

class Normaliser:
    def fit(self, X): raise NotImplementedError
    def transform(self, X): raise NotImplementedError
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMaxScaler(Normaliser):
    def fit(self, X):
        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)
    def transform(self, X):
        denom = self.xmax - self.xmin
        denom = np.where(denom == 0, 1, denom)
        return (X - self.xmin) / denom

class MeanNormalisation(Normaliser):
    def fit(self, X):
        self.mu = X.mean(axis=0)
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
        self.j = np.ceil(np.log10(np.where(max_abs == 0, 1, max_abs))).astype(int)
    def transform(self, X):
        return X / (10 ** self.j)

class RobustScaler(Normaliser):
    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        iqr = self.q3 - self.q1
        self.iqr = np.where(iqr == 0, 1, iqr)
    def transform(self, X):
        return (X - self.median) / self.iqr

class ZScoreStandardiser(Normaliser):
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
    def transform(self, X):
        return (X - self.mu) / self.std

class ModifiedZScore(Normaliser):
    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.mad = np.median(np.abs(X - self.median), axis=0)
        self.mad = np.where(self.mad == 0, 1, self.mad)
    def transform(self, X):
        return 0.6745 * (X - self.median) / self.mad

class ParetoScaling(Normaliser):
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.sqrt_std = np.where(self.std == 0, 1, np.sqrt(self.std))
    def transform(self, X):
        return (X - self.mu) / self.sqrt_std

class LogTransformation(Normaliser):
    def fit(self, X):
        self.c = np.where(X.min(axis=0) <= 0, np.abs(X.min(axis=0)) + 1, 0)
    def transform(self, X):
        return np.log(X + self.c)

class ReciprocalTransformation(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        safe = np.where(X == 0, np.finfo(float).eps, X)
        result = 1.0 / safe
        return np.clip(result, -1e10, 1e10)

class SquareRootTransformation(Normaliser):
    def fit(self, X):
        self.shift = np.where(X.min(axis=0) < 0, np.abs(X.min(axis=0)), 0)
    def transform(self, X):
        return np.sqrt(X + self.shift)

class BoxCoxLambdaHalf(Normaliser):
    def fit(self, X):
        self.shift = np.where(X.min(axis=0) <= 0, np.abs(X.min(axis=0)) + 1, 0)
    def transform(self, X):
        Xp = X + self.shift
        lam = 0.5
        return (np.power(Xp, lam) - 1) / lam

class YeoJohnsonLambdaHalf(Normaliser):
    def fit(self, X): pass
    def transform(self, X):
        lam = 0.5
        result = np.zeros_like(X, dtype=float)
        pos = X >= 0
        neg = ~pos
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
        X_clipped = np.clip(X, -500, 500)
        return 1.0 / (1.0 + np.exp(-X_clipped))

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
    def fit(self, X): pass
    def transform(self, X):
        Xs = X - X.max(axis=1, keepdims=True)
        ex = np.exp(Xs)
        return ex / ex.sum(axis=1, keepdims=True)

class QuantileNormalisationUniform(Normaliser):
    def fit(self, X):
        self.n_train = X.shape[0]
    def transform(self, X):
        n = X.shape[0]
        result = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = np.argsort(np.argsort(X[:, j]))
            result[:, j] = (ranks + 0.5) / n
        return result

class RankGauss(Normaliser):
    def fit(self, X):
        self.n_train = X.shape[0]
    def transform(self, X):
        n = X.shape[0]
        result = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = np.argsort(np.argsort(X[:, j]))
            u = (ranks + 0.5) / n
            u = np.clip(u, 1e-6, 1 - 1e-6)
            result[:, j] = np.sqrt(2) * erfinv(2 * u - 1)
        return result

NORMALISERS = {
    "1. Min-Max Scaling": MinMaxScaler(),
    "2. Mean Normalisation": MeanNormalisation(),
    "3. Max Absolute Scaling": MaxAbsoluteScaler(),
    "4. Decimal Scaling": DecimalScaling(),
    "5. Robust Scaling (IQR)": RobustScaler(),
    "6. Z-Score Standardisation": ZScoreStandardiser(),
    "7. Modified Z-Score": ModifiedZScore(),
    "8. Pareto Scaling": ParetoScaling(),
    "9. Log Transformation": LogTransformation(),
    "10. Reciprocal Transformation": ReciprocalTransformation(),
    "11. Square Root Transform": SquareRootTransformation(),
    "12. Box-Cox (λ=0.5)": BoxCoxLambdaHalf(),
    "13. Yeo-Johnson (λ=0.5)": YeoJohnsonLambdaHalf(),
    "14. Hyperbolic Tangent": HyperbolicTangent(),
    "15. Sigmoid / Logistic": SigmoidLogistic(),
    "16. L1 Normalisation": L1Normalisation(),
    "17. L2 Normalisation": L2Normalisation(),
    "18. Softmax Scaling": SoftmaxScaling(),
    "19. Quantile (Uniform)": QuantileNormalisationUniform(),
    "20. Rank-Gauss": RankGauss(),
}

def knn_predict(X_train, y_train, X_test, k):
    sq_train = (X_train ** 2).sum(axis=1)
    sq_test = (X_test ** 2).sum(axis=1)
    dists_sq = sq_test[:, None] + sq_train[None, :] - 2 * (X_test @ X_train.T)
    dists_sq = np.maximum(dists_sq, 0)
    preds = []
    for row_dists in dists_sq:
        nn_idx = np.argpartition(row_dists, min(k, len(row_dists)-1))[:k]
        votes = Counter(y_train[nn_idx])
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
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)

def kfold_cv_knn(X, y, k_values, n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    cv_accs = {k: [] for k in k_values}
    for fold_i in range(n_folds):
        val_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xvl, yvl = X[val_idx], y[val_idx]
        for k in k_values:
            preds = knn_predict(Xtr, ytr, Xvl, k)
            cv_accs[k].append(accuracy(yvl, preds))
    return {k: np.mean(v) for k, v in cv_accs.items()}


print("=" * 80)
print("STEP 2: EXPERIMENT - EVALUATING ALL METHODS")
print("=" * 80)
print()

K_VALUES = list(range(1, 31))
results = []

print("Evaluating Raw Data (baseline) …", end=" ", flush=True)
cv_raw = kfold_cv_knn(X_train, y_train, K_VALUES)
best_k = max(cv_raw, key=cv_raw.get)
y_pred = knn_predict(X_train, y_train, X_test, best_k)
acc_raw = accuracy(y_test, y_pred)
f1_raw = macro_f1(y_test, y_pred)
results.append({
    "Method": "Raw Data (baseline)",
    "Best k": best_k,
    "CV Accs": cv_raw,
    "Test Acc": acc_raw,
    "Macro F1": f1_raw,
    "Transformed": None,
})
print(f"k={best_k}  Acc={acc_raw:.4f}")

for name, norm in NORMALISERS.items():
    print(f"Evaluating {name} …", end=" ", flush=True)
    try:
        norm.fit(X_train)
        Xtr_n = norm.transform(X_train)
        Xte_n = norm.transform(X_test)
        Xtr_n = np.nan_to_num(Xtr_n, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_n = np.nan_to_num(Xte_n, nan=0.0, posinf=0.0, neginf=0.0)
        cv_scores = kfold_cv_knn(Xtr_n, y_train, K_VALUES)
        best_k = max(cv_scores, key=cv_scores.get)
        y_pred = knn_predict(Xtr_n, y_train, Xte_n, best_k)
        acc = accuracy(y_test, y_pred)
        f1 = macro_f1(y_test, y_pred)
        results.append({
            "Method": name,
            "Best k": best_k,
            "CV Accs": cv_scores,
            "Test Acc": acc,
            "Macro F1": f1,
            "Transformed": (Xtr_n, Xte_n),
        })
        print(f"k={best_k}  Acc={acc:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "Method": name, 
            "Best k": "N/A", 
            "CV Accs": {}, 
            "Test Acc": 0.0, 
            "Macro F1": 0.0,
            "Transformed": None,
        })

print()

print("=" * 80)
print("STEP 3: RESULTS & EXPORT")
print("=" * 80)
print()

df_results = pd.DataFrame([{
    "Method": r["Method"],
    "Best k": r["Best k"],
    "Test Acc": round(r["Test Acc"], 4),
    "Macro F1": round(r["Macro F1"], 4),
} for r in results])

df_results["Rank"] = df_results["Test Acc"].rank(ascending=False, method="min").astype(int)
df_results = df_results.sort_values("Rank").reset_index(drop=True)

print("MASTER RESULTS TABLE:")
print(df_results.to_string(index=False))
print()

best_method = df_results.iloc[0]["Method"]
worst_method = df_results.iloc[-1]["Method"]
print(f"Best  method : {best_method}")
print(f"Worst method : {worst_method}")
print()

# Save CSV
df_results.to_csv(RESULTS_CSV, index=False)
print(f"  Results saved to CSV: {os.path.abspath(RESULTS_CSV)}")
print()


print("=" * 80)
print("STEP 4: GENERATING VISUALIZATIONS FOR ALL 20 NORMALISERS")
print("=" * 80)
print()

FEAT_IDX = 0

def get_transformed_col(method_name, col_idx, use_transformed=None):
    """Get transformed feature column for visualization."""
    if method_name == "Raw Data (baseline)":
        return X_all[:, col_idx]
    
    if use_transformed is not None:
        Xtr_n, Xte_n = use_transformed
        combined = np.concatenate([Xtr_n, Xte_n], axis=0)
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        return combined[:, col_idx]
    
    norm = NORMALISERS.get(method_name)
    if norm is None:
        return X_all[:, col_idx]
    norm.fit(X_train)
    Xtr_n = norm.transform(X_train)
    Xte_n = norm.transform(X_test)
    combined = np.concatenate([Xtr_n, Xte_n], axis=0)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    return combined[:, col_idx]

# Create individual plots for each normalisation method
plt.style.use('seaborn-v0_8-darkgrid')

for idx, result in enumerate(results):
    method_name = result["Method"]
    test_acc = result["Test Acc"]
    best_k = result["Best k"]
    
    # Create figure with 2 subplots: histogram + CV accuracy curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Feature distribution histogram
    vals = get_transformed_col(method_name, FEAT_IDX, result.get("Transformed"))
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    vals_clipped = np.clip(vals, lo, hi)
    
    ax1.hist(vals_clipped, bins=40, color='steelblue', edgecolor='white', lw=0.5, alpha=0.85)
    ax1.set_title(f"Feature {FEAT_IDX} Distribution\n{method_name}", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Normalized Value")
    ax1.set_ylabel("Frequency")
    ax1.grid(alpha=0.3)
    
    # Plot 2: CV Accuracy vs k
    cv_accs = result["CV Accs"]
    if cv_accs:
        ks = sorted(cv_accs.keys())
        accs = [cv_accs[k] for k in ks]
        ax2.plot(ks, accs, color='seagreen', lw=2.5, marker='o', markersize=4)
        ax2.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
        ax2.set_title(f"CV Accuracy vs k\nTest Acc: {test_acc:.4f}", fontsize=11, fontweight='bold')
        ax2.set_xlabel("k (number of neighbours)")
        ax2.set_ylabel("CV Accuracy")
        ax2.grid(alpha=0.3)
        ax2.legend()
    
    fig.suptitle(f"{idx}. {method_name} | Test Accuracy: {test_acc:.4f} | Macro F1: {result['Macro F1']:.4f}", 
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save with safe filename
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-").replace(".", "")
    filename = f"{idx:02d}_{safe_name}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f" {idx+1}/21 : {method_name:40s} → {filename}")

    
# --- BEGIN: Extra Visualizations for Raw, Best, Worst ---

    # 1. Feature distribution comparison (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ["Raw Data (baseline)", best_method, worst_method]
    titles = ["Raw Data (baseline)", f"Best: {best_method}", f"Worst: {worst_method}"]
    for i, (method, title) in enumerate(zip(methods, titles)):
        vals = get_transformed_col(method, FEAT_IDX)
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        vals_clipped = np.clip(vals, lo, hi)
        axes[i].hist(vals_clipped, bins=40, color='steelblue', edgecolor='white', lw=0.5, alpha=0.85)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Normalized Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(alpha=0.3)
    plt.suptitle(f"Feature {FEAT_IDX} Distribution: Raw vs Best vs Worst", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_distribution_raw_best_worst.png"), dpi=100, bbox_inches='tight')
    plt.close()     
    # 2. Overlayed CV accuracy vs k curves
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for method, color, label in zip(methods, colors, titles):
        result = next(r for r in results if r["Method"] == method)
        cv_accs = result["CV Accs"]
        if cv_accs:
            ks = sorted(cv_accs.keys())
            accs = [cv_accs[k] for k in ks]
            plt.plot(ks, accs, label=label, lw=2.5, marker='o', markersize=4, color=color)
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("CV Accuracy")
    plt.title("CV Accuracy vs k: Raw vs Best vs Worst", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "cv_accuracy_vs_k_raw_best_worst.png"), dpi=100, bbox_inches='tight')
    plt.close()
    # --- END: Extra Visualizations for Raw, Best, Worst ---
        

print()
print(f"  All visualizations saved to: {os.path.abspath(IMAGES_DIR)}")
print()

print("=" * 80)
print("✓ Q2 COMPLETE!")
print("=" * 80)
print()
print("SUMMARY:")
print()
print(f"  Best Method          : {best_method}")
print(f"  Best Test Acc        : {df_results.iloc[0]['Test Acc']:.4f}")
print(f"  Best Macro F1        : {df_results.iloc[0]['Macro F1']:.4f}")
print()
print(f"  Worst Method         : {worst_method}")
print(f"  Worst Test Acc       : {df_results.iloc[-1]['Test Acc']:.4f}")
print(f"  Worst Macro F1       : {df_results.iloc[-1]['Macro F1']:.4f}")
print()
print("=" * 80)