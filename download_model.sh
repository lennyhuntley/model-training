#!/bin/bash

#!/bin/bash

# This script downloads the latest trained model and the labels dataset.
# It performs two main actions:
# 1. Downloads the latest TensorFlow SavedModel from GCS and extracts it into `assets/models/birds/`.
# 2. Downloads the `nabirds_mini_stratified.zip` dataset and places it in the same directory.

# The script will automatically install required Python packages (`google-cloud-storage`, `requests`).

# --- Configuration ---
# This should match the base name of the model from the training script.
MODEL_NAME="mobilenetv2_train_base_True"

# --- Execution ---
echo "Starting download process for model: $MODEL_NAME"

python3 download_model.py --model_name=$MODEL_NAME

echo "\nDownload process finished. Files are in assets/models/birds/"


