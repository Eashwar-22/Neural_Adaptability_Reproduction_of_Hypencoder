
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

# Style Configuration
THEME_COLOR = "#A51E37"
sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 300
})

def analyze_fingerprints():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "./checkpoints/hypencoder.6_layer_full_real_opt"
    output_dir = "msc_thesis/images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model and tokenizer...")
    model_dual = HypencoderDualEncoder.from_pretrained(checkpoint_path, use_safetensors=True).to(device)
    model = model_dual.query_encoder
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()

    # Define Categories and Queries (32 total, 8 archetypes x 4)
    queries = {
        "Medical": [
            "symptoms of cardiac arrest", 
            "how long do flu symptoms last", 
            "causes of high blood pressure", 
            "treatment for common cold"
        ],
        "Technical": [
            "how to install python on ubuntu", 
            "center a div in css", 
            "connect to wifi on raspberry pi", 
            "git merge vs rebase"
        ],
        "Entity": [
            "who is the CEO of apple", 
            "birthplace of albert einstein", 
            "who wrote the great gatsby", 
            "current governor of california"
        ],
        "Factoid": [
            "distance from earth to moon", 
            "boiling point of water at sea level", 
            "capital of australia", 
            "number of states in the us"
        ],
        "Definition": [
            "define photosynthesis", 
            "meaning of ubiquitous", 
            "what is a black hole", 
            "definition of irony"
        ],
        "Narrative": [
            "From what I remember it involved 3-4 astronauts. One stays in orbit (a black man), while the others go to the surface of this planet. The ones on the surface are there to retrieve data (from probes perhaps?). The astronauts on the surface only have a limited time to collect this data and get back to the ship in orbit. I think something happens to shorten their time there (a storm maybe?). During this scene they were surrounded by water if I remember correctly. The female (possibly the crew leader?) keeps saying she can finish the data collection in time because it’ll be ages before they can get another chance. She says this despite the warnings from her comrades to just leave. Well, they miss their chance to get back to the ship in orbit. Now the kicker is that the time on the planet’s surface is vastly different to the time in orbit. For instance, on the surface perhaps only a few hours pass, but in orbit several years pass. I think it ends up being something like 20+ years to the man in orbit before the surface crew gets back. Anyone know what this movie is? I watched it in the last 3 years, but I streamed it so the movie itself could be older. Although I don’t think it would be more then say ten years old. It’s in colour, an American film in English.",
            "An old movie seen during 1990s on TV. I think it was in english and in color . The airplane of a family fall in a strange area. there is a scene of introducing disgusting food; eyes in soup. And a scene of putting a man in metal cage then in fire.",
            "I randomly remembered this movie (or maybe an episode of a tv show) where a girl came home late/the next day from prom and her mom asked where she was and she lied and said she got drunk by mixing red and white wine. I think the real reason she didn’t come home was like, she met her real dad or something? For some reason I remember it as Julia Stiles but i looked at imdb and nothing on her filmology page rings any bells. Unfortunately I don’t remember much about the date.",
            "I hope someone can help me track down this movie. I may have some details mixed up, or combined with another drama, so if you have any suggestions at all, please share them. There is a Caucasian man, who is a little person, who is successful in his working life but is lonely in his private life. He may be a writer or editor, or some such professional. I think part of the plot is that he might have trouble making friends because some people are prejudiced against his size. Sometimes he abuses alcohol because of this. He re-connects with a woman he knew when he was a child, even though she used to tease him back when they were children. They become friends and eventually fall in love. She is a regular height woman, also Caucasian, slim and with red hair. In the last scene of the movie, they are dancing on a balcony or rooftop and she is wearing a white dress. Maybe it’s their wedding? I have been through the filmographies of all the male Caucasian little people actors I can think of with no luck. My Googlefu has failed me as well. If you have any ideas at all, no matter how far fetched, please share them. Thank you in advance for any suggestions you may have."
        ],
        "Keyword": [
            "weather", 
            "amazon", 
            "google maps", 
            "translator"
        ],
        "Temporal": [
            "when did world war 2 end", 
            "date of last lunar eclipse", 
            "next olympics location", 
            "who won the super bowl 2024"
        ]
    }
    
    cat_names = list(queries.keys())
    flat_queries = []
    for q_list in queries.values():
        flat_queries.extend(q_list)

    print(f"Processing {len(flat_queries)} queries...")
    inputs = tokenizer(flat_queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_net = output.representation 

    # Extract Weights
    w_0 = q_net.layers[0].linear.weight.reshape(len(flat_queries), -1)
    w_last = q_net.layers[-1].weight.reshape(len(flat_queries), -1)

    def get_sim_matrix(tensors):
        t = F.normalize(tensors, p=2, dim=1)
        return torch.mm(t, t.t()).cpu().numpy()

    sim_0 = get_sim_matrix(w_0)
    sim_last = get_sim_matrix(w_last)
    
    # Compute Residuals for Final Layer
    w_mean = torch.mean(w_last, dim=0, keepdim=True)
    w_res = w_last - w_mean
    sim_res = get_sim_matrix(w_res)

    def plot_refined_heatmap(matrix, filename, vmin=None, vmax=None):
        plt.figure(figsize=(8, 7))
        
        ax = sns.heatmap(matrix, cmap="magma", square=True, cbar_kws={"shrink": 0.8},
                        vmin=vmin, vmax=vmax)
        
        # Remove individual ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        num_queries = matrix.shape[0]
        num_categories = len(cat_names)
        queries_per_cat = num_queries // num_categories
        
        # Add category labels centered in the blocks
        for j, cat in enumerate(cat_names):
            pos = (queries_per_cat / 2) + j * queries_per_cat
            # X-axis (Bottom)
            ax.text(pos, num_queries + 0.5, cat, ha='center', va='top', fontsize=10, rotation=45)
            # Y-axis (Left)
            ax.text(-0.5, pos, cat, ha='right', va='center', rotation=0, fontsize=10)
            
            # Add grid lines to separate categories
            if j > 0:
                ax.axvline(j * queries_per_cat, color='white', lw=1.0, alpha=0.5)
                ax.axhline(j * queries_per_cat, color='white', lw=1.0, alpha=0.5)
        
        # End lines
        # ax.axvline(num_queries, color='white', lw=1.5, alpha=0.3)
        # ax.axhline(num_queries, color='white', lw=1.5, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

    # Plot each separately
    plot_refined_heatmap(sim_0, "qnet_layer1_sim.png")
    plot_refined_heatmap(sim_last, "qnet_final_sim.png", vmin=0.9, vmax=1.0)
    plot_refined_heatmap(sim_res, "qnet_residual_sim.png", vmin=-0.3, vmax=1.0)

if __name__ == "__main__":
    analyze_fingerprints()
