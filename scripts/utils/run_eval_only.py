
import fire
from pathlib import Path
from hypencoder_cb.utils.eval_utils import load_standard_format_as_run, calculate_metrics_to_file
from hypencoder_cb.utils.data_utils import load_qrels_from_ir_datasets

def main(retrieval_path, output_dir, ir_dataset_name="msmarco-passage/dev/small"):
    print(f"Running evaluation for {retrieval_path}")
    print(f"Dataset: {ir_dataset_name}")
    print(f"Output: {output_dir}")
    
    # 1. Load Run
    print("Loading run...")
    run = load_standard_format_as_run(retrieval_path)
    
    # 2. Load Qrels
    print(f"Loading qrels from {ir_dataset_name}...")
    qrels = load_qrels_from_ir_datasets(ir_dataset_name)
    
    # 3. Calculate metrics
    print("Calculating metrics...")
    output_dir = Path(output_dir)
    calculate_metrics_to_file(
        run=run,
        qrels=qrels,
        output_folder=output_dir,
        metric_names=["nDCG@10", "P@10", "P@5", "R@10", "R@1000", "MRR@10", "MRR"]
    )
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)
