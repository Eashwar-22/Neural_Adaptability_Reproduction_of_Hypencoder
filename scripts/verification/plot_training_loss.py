import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(json_path, output_path):
    print(f"Loading history from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data.get('log_history', [])
    steps = []
    losses = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            
    print(f"Found {len(steps)} data points.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve: Retrained Hypencoder')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add annotation for final loss
    if steps:
        final_loss = losses[-1]
        final_step = steps[-1]
        plt.annotate(f'Final: {final_loss:.4f}', 
                     xy=(final_step, final_loss), 
                     xytext=(final_step, final_loss + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    
    plot_loss(args.json_path, args.output_path)
