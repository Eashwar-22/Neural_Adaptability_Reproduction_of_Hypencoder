
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse

def main(args):
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Enable Half Precision for speed/memory if on CUDA
    if torch.cuda.is_available():
        model.half()

def main(args):
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Enable Half Precision for speed/memory if on CUDA
    if torch.cuda.is_available():
        model.half()

    print(f"Reading from {args.input_file}")
    
    batch_size = args.batch_size
    buffer = []

    with open(args.input_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        for i, line in tqdm(enumerate(f_in)):
            buffer.append(json.loads(line))
            
            if len(buffer) >= batch_size:
                process_buffer(buffer, model, tokenizer, device, f_out)
                buffer = []
        
        if buffer:
            process_buffer(buffer, model, tokenizer, device, f_out)

def process_buffer(buffer, model, tokenizer, device, f_out):
    try:
        all_pairs = []
        pair_counts = [] # Number of pairs per item (query)
        
        for item in buffer:
            query_text = item['query']['content']
            # Form pairs: [(query, doc1), (query, doc2), ...]
            pairs = [[query_text, x['content']] for x in item['items']]
            all_pairs.extend(pairs)
            pair_counts.append(len(pairs))
            
        # Tokenize ALL pairs in batch
        with torch.no_grad():
            inputs = tokenizer(all_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().tolist()
        
        # Distribute scores back
        current_idx = 0
        for i, count in enumerate(pair_counts):
            item_scores = scores[current_idx : current_idx + count]
            current_idx += count
            
            # Update item
            for j, item_obj in enumerate(buffer[i]['items']):
                item_obj['score'] = item_scores[j]
            
            f_out.write(json.dumps(buffer[i]) + '\n')
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
