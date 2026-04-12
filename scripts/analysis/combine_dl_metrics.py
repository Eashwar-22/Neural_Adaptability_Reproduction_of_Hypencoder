import os
import pandas as pd
import sys

def load_metrics(path):
    if not os.path.exists(path):
        return None
    try:
        # Read the csv file (metrics/aggregated_metrics.txt is csv format)
        df = pd.read_csv(path)
        return df.iloc[0] # Return first row as series
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    base_dir = "/home/eickhoff/esx510/hypencoder-paper/outputs/inference/hypencolbert"
    dl19_path = os.path.join(base_dir, "dl19_results", "metrics", "aggregated_metrics.txt")
    dl20_path = os.path.join(base_dir, "dl20_results", "metrics", "aggregated_metrics.txt")

    print(f"Reading DL19 from: {dl19_path}")
    print(f"Reading DL20 from: {dl20_path}")

    m19 = load_metrics(dl19_path)
    m20 = load_metrics(dl20_path)

    if m19 is None:
        print("DL 19 results not found yet.")
        return
    if m20 is None:
        print("DL 20 results not found (unexpected).")
        return

    # Combine
    # We want average of nDCG@10, R@1000, MRR (if exists)
    # Check common keys
    keys = ["nDCG@10", "R@1000"]
    if "MRR" in m19 and "MRR" in m20:
        keys.append("MRR")
    
    combined = {}
    for k in keys:
        if k in m19 and k in m20:
            val = (m19[k] + m20[k]) / 2.0
            combined[k] = val
            print(f"{k}: (DL19={m19[k]:.4f} + DL20={m20[k]:.4f}) / 2 = {val:.4f}")
        else:
            print(f"Missing {k} in one of the files.")

    print("\n Markdown Table Row:")
    print(f"|**TREC DL '19 & '20**|nDCG@10| - | - | - | {combined.get('nDCG@10', 0.0):.3f} | ...")
    print(f"||R@1000| - | - | - | {combined.get('R@1000', 0.0):.3f} | ...")

if __name__ == "__main__":
    main()
