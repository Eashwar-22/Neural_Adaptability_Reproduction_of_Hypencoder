import argparse
from datasets import load_dataset
import os
import glob

def main():
    parser = argparse.ArgumentParser(description="Parallel JSONL to HF Dataset Converter (Split)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing data chunks")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the final dataset")
    parser.add_argument("--num_workers", type=int, default=30, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Use HF cache dir
    cache_dir = os.environ.get("HF_HOME", "${HYPENCODER_CACHE:-./cache}")
    
    # Find all chunk files
    data_files = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    print(f"Found {len(data_files)} chunks in {args.input_dir}. Loading with num_proc={args.num_workers}...")
    
    # load_dataset with multiple files automagically uses multiple processes if num_proc > 1
    dataset = load_dataset(
        "json", 
        data_files=data_files, 
        split="train", 
        num_proc=args.num_workers,
        cache_dir=cache_dir
    )
    
    print(f"Loaded {len(dataset)} examples. Saving to disk at {args.output_dir}...")
    dataset.save_to_disk(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
