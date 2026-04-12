#!/bin/bash
#SBATCH --job-name="dl_assets"
#SBATCH --partition=cpu-ferranti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_assets_%j.log

source ~/.bashrc
conda activate hype_env

# Move to project root
cd .

echo "Starting download job on $(hostname)"
chmod +x ./download_assets.sh
./download_assets.sh

echo "Job complete."
