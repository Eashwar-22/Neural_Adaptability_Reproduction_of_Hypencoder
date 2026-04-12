#!/bin/bash
# Download assets for inference (Embeddings & Neighbor Graph)

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found. Please ensure the conda environment is active."
    exit 1
fi

mkdir -p assets
cd assets

echo "--- Downloading Hypencoder 6-Layer Encoded Items (Folder) ---"
# Google Drive Folder ID for "hypencoder.6_layer.encoded_items"
gdown --folder 1htoVx8fAVm-4ZfdssAXdw-_D-Kzs59dx -O hypencoder.6_layer.encoded_items

echo "--- Downloading Neighbor Graph (File) ---"
# Google Drive File ID for "hypencoder.6_layer.neighbor_graph"
gdown 1EhKuGxaFI51DDSDqsoAwiYRs1IdZATrk -O hypencoder.6_layer.neighbor_graph

echo "Download complete. Assets are in the 'assets' directory."
