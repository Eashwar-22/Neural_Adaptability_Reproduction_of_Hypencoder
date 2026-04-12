
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def main():
    print("Loading Cross-Encoder Teacher (transformers)...")
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Pre-load student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    OUTPUT_FILE = 'data/crossencoder_distill_1k.jsonl'
    
    samples = []
    print("Reading triples...")
    with open('data/triples.train.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            d = json.loads(line)
            if d['items'][0]['content'] == d['items'][1]['content']:
                continue
            samples.append(d)

    print(f"Scoring {len(samples)} samples with Cross-Encoder...")
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for s in tqdm(samples):
            q_text = s['query']['content']
            pos_text = s['items'][0]['content']
            neg_text = s['items'][1]['content']
            
            # Prepare pairs
            pairs = [[q_text, pos_text], [q_text, neg_text]]
            
            # Predict
            with torch.no_grad():
                features = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
                scores = model(**features).logits
                # Verify shape/logic: CrossEncoder usually outputs a single score per pair
                # ms-marco-MiniLM-L-6-v2 outputs 1 logit? Or 2? 
                # According to HF, it's 1 label usually for regression or 2 for classification.
                # Let's assume it returns a scalar (1 unit output layer) or we take index 1.
                # Just checking: ms-marco-MiniLM-L-6-v2 is trained for CE loss usually?
                # Actually, cross-encoder/ms-marco-MiniLM-L-6-v2 outputs a single score (logit).
                
                if scores.shape[1] == 1:
                    pos_score = scores[0].item()
                    neg_score = scores[1].item()
                else:
                    # If 2 classes (Binary), usually index 1 is 'relevant'.
                    pos_score = scores[0][1].item()
                    neg_score = scores[1][1].item()
            
            # Tokenize for Student
            q_tok = student_tokenizer(q_text, truncation=True, max_length=512)['input_ids']
            p_tok = student_tokenizer(pos_text, truncation=True, max_length=512)['input_ids']
            n_tok = student_tokenizer(neg_text, truncation=True, max_length=512)['input_ids']
            
            if p_tok == n_tok:
                continue

            record = {
                'query': {'tokenized_content': q_tok},
                'items': [
                    {'tokenized_content': p_tok, 'score': float(pos_score)},
                    {'tokenized_content': n_tok, 'score': float(neg_score)}
                ]
            }
            f_out.write(json.dumps(record) + '\n')

    print(f"Done! Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
