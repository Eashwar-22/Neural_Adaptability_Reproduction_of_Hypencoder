
import re
import matplotlib.pyplot as plt
import datetime

# File paths
log_file = "logs/FINAL_HYPENCOLBERT.log"
output_image = "logs/training_loss_hypencolbert.png"

steps = []
losses = []

# Regex pattern to find lines like: {'loss': 1.664, ..., 'epoch': 66.24}
# We need to track steps manually because steps aren't always explicitly in the JSON line,
# but we know logging_steps=50. However, the log is a resumed run starting at epoch 66 (~65k steps).
# Let's try to extract steps if available, or infer them.
# Looking at previous log output:
# {'loss': 1.664, 'learning_rate': ..., 'epoch': 66.24}
# It doesn't have 'step'. We'll use 'epoch' for X-axis or infer step count.

epochs = []

print("Parsing log file...")
with open(log_file, 'r') as f:
    for line in f:
        if "{'loss':" in line:
            try:
                # Extract dictionary part
                dict_str = line.strip().replace("'", '"')
                # Simple parsing or regex
                loss_match = re.search(r"'loss':\s*([0-9.]+)", line)
                epoch_match = re.search(r"'epoch':\s*([0-9.]+)", line)
                
                if loss_match and epoch_match:
                    losses.append(float(loss_match.group(1)))
                    epochs.append(float(epoch_match.group(1)))
            except Exception as e:
                print(f"Skipping line: {line[:50]}... Error: {e}")

if not losses:
    print("No loss data found!")
    exit(1)

print(f"Found {len(losses)} data points.")

# Determine steps approx (steps per epoch = 250,000 / 254 ~ 984 steps/epoch?)
# Actually simpler to just plot Logged Steps (0, 1, 2...) or Epochs.
# Let's plot against Epochs as it's the raw data we have.

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Training Loss', color='blue', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss - Hypencoder (unpooled)\n(Stagnated at ~1.0)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save plot
plt.savefig(output_image)
print(f"Plot saved to {output_image}")
