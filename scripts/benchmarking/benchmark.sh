#!/bin/bash
#SBATCH --job-name="bench_hype"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=benchmark_output.txt

source ~/.bashrc
conda activate hype_env
python benchmark_speed.py
