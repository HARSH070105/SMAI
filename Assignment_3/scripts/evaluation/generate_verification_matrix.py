import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "verification")

models = ["base", "baseline_lora", "ensemble_lora", "custom_ensemble_lora"]
prompts = ["baseline", "ensemble", "custom_ensemble"]

# Initialize matrices
matrix_eer = {m: {p: np.nan for p in prompts} for m in models}
matrix_auc = {m: {p: np.nan for p in prompts} for m in models}
matrix_tar = {m: {p: np.nan for p in prompts} for m in models}

# Regex to find the metrics
eer_pattern = re.compile(r"Equal Error Rate \(EER\):\s+([\d\.]+)%")
auc_pattern = re.compile(r"ROC AUC Score:\s+([\d\.]+)")
tar_pattern = re.compile(r"TAR @ 1\.0% FAR:\s+([\d\.]+)%")

print(f"Scanning verification reports directory: {REPORTS_DIR}\n")

for m in models:
    for p in prompts:
        report_path = os.path.join(REPORTS_DIR, m, f"{p}_prompts", "verification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                content = f.read()
                
                eer_match = eer_pattern.search(content)
                if eer_match:
                    matrix_eer[m][p] = float(eer_match.group(1))
                    
                auc_match = auc_pattern.search(content)
                if auc_match:
                    matrix_auc[m][p] = float(auc_match.group(1))
                    
                tar_match = tar_pattern.search(content)
                if tar_match:
                    matrix_tar[m][p] = float(tar_match.group(1))

# --- Format and print textual report ---
def get_matrix_string(title, matrix_dict, is_percent=False):
    col_width = 22
    out = f"### {title}\n\n"
    
    header = "| " + "Model \\ Prompt".ljust(col_width) + " |"
    for p in prompts:
        header += f" {p.ljust(col_width)} |"
    out += header + "\n"

    separator = "|" + "-" * (col_width + 2) + "|"
    for _ in prompts:
        separator += "-" * (col_width + 2) + "|"
    out += separator + "\n"

    for m in models:
        row = f"| {m.ljust(col_width)} |"
        for p in prompts:
            val = matrix_dict[m][p]
            if np.isnan(val):
                val_str = "N/A"
            else:
                val_str = f"{val:.2f}%" if is_percent else f"{val:.4f}"
            row += f" {val_str.ljust(col_width)} |"
        out += row + "\n"
    return out + "\n"

full_report = ""
full_report += get_matrix_string("Equal Error Rate (EER) Matrix [Lower is Better]", matrix_eer, is_percent=True)
full_report += get_matrix_string("ROC AUC Score Matrix [Higher is Better]", matrix_auc, is_percent=False)
full_report += get_matrix_string("True Accept Rate (TAR) @ 1.0% FAR Matrix [Higher is Better]", matrix_tar, is_percent=True)

print(full_report)

# Save textual report to markdown file
txt_save_path = os.path.join(REPORTS_DIR, "overall_verification_report.md")
with open(txt_save_path, "w") as f:
    f.write("# Overall Verification Metrics\n\n")
    f.write(full_report)
print(f"Saved textual report to: {txt_save_path}")

# --- Plotting Heatmaps ---
num_eer = np.array([[matrix_eer[m][p] for p in prompts] for m in models])
num_auc = np.array([[matrix_auc[m][p] for p in prompts] for m in models])
num_tar = np.array([[matrix_tar[m][p] for p in prompts] for m in models])

fig, axes = plt.subplots(1, 3, figsize=(26, 8))

# 1. EER Heatmap (Lower is better, use Reds)
sns.heatmap(num_eer, annot=True, fmt=".2f", cmap="Reds", 
            xticklabels=prompts, yticklabels=models, 
            cbar_kws={'label': 'EER (%)'}, ax=axes[0], annot_kws={"size": 18})
for t in axes[0].texts:
    if t.get_text() != "nan": t.set_text(t.get_text() + "%")
axes[0].set_title("Equal Error Rate (EER)\n[Lower is Better]", fontsize=16, pad=15)
axes[0].set_xlabel("Prompt Type", fontsize=12)
axes[0].set_ylabel("Model Type", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# 2. ROC AUC Heatmap (Higher is better)
sns.heatmap(num_auc, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=prompts, yticklabels=models, 
            cbar_kws={'label': 'ROC AUC'}, ax=axes[1], annot_kws={"size": 18})
axes[1].set_title("ROC AUC Score\n[Higher is Better]", fontsize=16, pad=15)
axes[1].set_xlabel("Prompt Type", fontsize=12)
axes[1].set_ylabel("")
axes[1].tick_params(axis='x', rotation=45)

# 3. TAR Heatmap (Higher is better)
sns.heatmap(num_tar, annot=True, fmt=".2f", cmap="YlGnBu", 
            xticklabels=prompts, yticklabels=models, 
            cbar_kws={'label': 'TAR (%)'}, ax=axes[2], annot_kws={"size": 18})
for t in axes[2].texts:
    if t.get_text() != "nan": t.set_text(t.get_text() + "%")
axes[2].set_title("TAR @ 1.0% FAR\n[Higher is Better]", fontsize=16, pad=15)
axes[2].set_xlabel("Prompt Type", fontsize=12)
axes[2].set_ylabel("")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()

save_path = os.path.join(REPORTS_DIR, "verification_metrics_heatmap.png")
fig.savefig(save_path, dpi=200)
print(f"Saved visualization to: {save_path}")
