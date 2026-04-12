import os
import sys

def create_split_streaming(input_file, val_file, train_file, num_val_samples=2000):
    print(f"Counting lines in {input_file}...")
    # Efficient line counting
    total_lines = 0
    with open(input_file, 'rb') as f:
        for _ in f:
            total_lines += 1
    
    print(f"Total samples: {total_lines}")
    
    if total_lines <= num_val_samples:
        print("Error: Not enough data to create a split.")
        return

    # Determine split point (take last N for validation)
    split_index = total_lines - num_val_samples
    
    print(f"Splitting at line {split_index}...")
    print(f"Validation target: {val_file}")
    print(f"Training target: {train_file}")
    
    current_line = 0
    
    with open(input_file, 'r') as fin, \
         open(train_file, 'w') as ftrain, \
         open(val_file, 'w') as fval:
         
        for line in fin:
            if current_line < split_index:
                ftrain.write(line)
            else:
                fval.write(line)
            current_line += 1
            
            if current_line % 1000000 == 0:
                print(f"Processed {current_line} lines...", end='\r')

    print(f"\nDone! Created:")
    print(f" - Train: {split_index} samples")
    print(f" - Val: {num_val_samples} samples")
    print("\nIMPORTANT: Delete any .dataset cache directories before retraining!")

if __name__ == "__main__":
    SOURCE_FILE = "data/triples.train.jsonl"
    VAL_FILE = "data/triples.val.jsonl"
    TRAIN_FILE = "data/triples.train.split.jsonl" # Using a NEW file first to be safe, then can rename
    
    # Check if we should overwrite in place?
    # User script had TRAIN_FILE overwrite SOURCE_FILE.
    # To be safe, let's write to a temporary file first, then user can swap if happy.
    # Or just overwrite if that was the intent. 
    # The user's previous script overwrote it.
    # Given the disk size (96GB), verifying free space is good, but overwriting in-place is hard with streaming without temp file.
    # We MUST write to a separate file first.
    
    # Let's confirm source exists
    if not os.path.exists(SOURCE_FILE):
        print(f"File {SOURCE_FILE} not found.")
        sys.exit(1)
        
    # Run split
    # writing to triples.train.jsonl.new
    TRAIN_FILE_NEW = "data/triples.train.new.jsonl"
    create_split_streaming(SOURCE_FILE, VAL_FILE, TRAIN_FILE_NEW)
    
    print("Renaming new training file to overwrite original...")
    os.rename(TRAIN_FILE_NEW, SOURCE_FILE)
    print("Original file overwritten. Validation split created.")
