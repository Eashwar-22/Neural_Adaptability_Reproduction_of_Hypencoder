
import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main(args):
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    print(f"Reading from {args.input_file}")
    
    instruction = "Represent this sentence for searching relevant passages: "
    
    batch_size = args.batch_size
    buffer = []

    with open(args.input_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        for i, line in tqdm(enumerate(f_in)):
            buffer.append(json.loads(line))
            
            if len(buffer) >= batch_size:
                process_buffer(buffer, model, tokenizer, device, instruction, f_out)
                buffer = []
        
        # Process remaining
        if buffer:
            process_buffer(buffer, model, tokenizer, device, instruction, f_out)

def process_buffer(buffer, model, tokenizer, device, instruction, f_out):
    try:
        # Prepare inputs
        queries = [instruction + item['query']['content'] for item in buffer]
        
        # Flatten docs and keep track of counts to reconstruction
        all_docs = []
        doc_counts = []
        for item in buffer:
            docs = [x['content'] for x in item['items']]
            all_docs.extend(docs)
            doc_counts.append(len(docs))
            
        # Encode Queries
        encoded_queries = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to(device)
        
        # Encode Docs
        # If too many docs, we might need to batch this sub-part too to avoid OOM, 
        # but handled by main batch_size usually.
        encoded_docs = tokenizer(all_docs, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Query Embeddings
            q_out = model(**encoded_queries)
            q_emb = q_out.last_hidden_state[:, 0]
            q_emb = F.normalize(q_emb, p=2, dim=1)
            
            # Doc Embeddings
            d_out = model(**encoded_docs)
            d_emb = d_out.last_hidden_state[:, 0]
            d_emb = F.normalize(d_emb, p=2, dim=1)
            
        # Compute scores
        # We need to compute scores for each query vs its own docs.
        # q_emb: [B, Dim]
        # d_emb: [TotalDocs, Dim]
        
        # Slicing d_emb based on counts
        current_idx = 0
        for i, count in enumerate(doc_counts):
            q_vec = q_emb[i].unsqueeze(0) # [1, Dim]
            d_vecs = d_emb[current_idx : current_idx + count] # [Count, Dim]
            current_idx += count
            
            # Dot product
            scores = torch.mm(q_vec, d_vecs.transpose(0, 1)).squeeze(0).cpu().tolist()
            
            # Update item
            for j, item_obj in enumerate(buffer[i]['items']):
                item_obj['score'] = scores[j]
            
            # Write immediately
            f_out.write(json.dumps(buffer[i]) + '\n')
            
    except Exception as e:
        print(f"Error processing batch: {e}")
        # In case of error, maybe try to process one by one or just skip batch? 
        # Skipping batch is safer for now to keep running.
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)
