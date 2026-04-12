#!/bin/bash
#SBATCH --job-name="colbert_base"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/colbert_baseline_%j.log

source ~/.bashrc
# Activate the dedicated ColBERT environment
source colbert_env/bin/activate

echo "Starting ColBERT Baseline run on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Verify faiss and numpy
python -c "import numpy; import faiss; print(f'Numpy: {numpy.__version__}, Faiss: {faiss.__version__}')"

# Run the python script
# We explicitly set the python executable from the venv just to be sure
colbert_env/bin/python scripts/run_colbert_baseline.py

echo "Job finished."
