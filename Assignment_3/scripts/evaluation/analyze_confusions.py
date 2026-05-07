import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Analyze a confusion matrix for misclassifications")
    parser.add_argument("--report_path", type=str, required=True, 
                        help="Path to the report directory (e.g. reports/base/baseline_prompts) or the confusion_matrix.csv file itself")
    args = parser.parse_args()

    # If the user passed a directory, look for the csv inside it
    if os.path.isdir(args.report_path):
        csv_file = os.path.join(args.report_path, "confusion_matrix.csv")
    else:
        csv_file = args.report_path

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        print("Note: The text classification_report.txt does not contain raw confusion pairs.")
        print("Please re-run evaluate.py to generate 'confusion_matrix.csv'.")
        return

    print("=" * 70)
    print(f"Analyzing Confusions in: {csv_file}")
    print("-" * 70)
    
    df = pd.read_csv(csv_file, index_col=0)
    
    confusions = []
    for true_class in df.index:
        for pred_class in df.columns:
            if true_class != pred_class:
                count = df.loc[true_class, pred_class]
                if count > 0:
                    confusions.append((true_class, pred_class, count))
    
    # Sort by frequency of confusion (descending)
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    if not confusions:
        print("  No confusions found! 100% accuracy.")
    else:
        for true_cls, pred_cls, count in confusions:
            print(f"  '{true_cls}' got confused with '{pred_cls}' -> {count} times")
    print("\n")

if __name__ == "__main__":
    main()
