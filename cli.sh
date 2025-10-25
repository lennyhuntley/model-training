
# List of prebuilt containers for training
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers

export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="know_now_training_job_$UUID"
export MACHINE_TYPE="n1-standard-4"
export REPLICA_COUNT=1
export EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-16.py310:latest"
export PYTHON_PACKAGE_URI=gs://know-now-app-trainer-lh/know-now-app-trainer.tar.gz
export PYTHON_MODULE="trainer.task"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export ACCELERATOR_COUNT=1
export GCP_REGION="us-central1" # Adjust region based on you approved quotas for GPUs

# Change the number of epochs
# Set the command-line arguments for the new fine-tuning script.
# Ensure WANDB_KEY is set in your environment: export WANDB_KEY='your_key_here'
export GCS_DATA_DIR="gs://kaggle_nabirds_data/nabirds_preprocessed"
export CMDARGS="--gcs_data_dir=$GCS_DATA_DIR,--percent_to_use=1.0,--epochs_warmup=2,--epochs_finetune=10,--batch_size=64,--lr_warmup=1e-3,--lr_fine=3e-5,--wandb_key=$WANDB_KEY"
# Run training with GPU
gcloud ai custom-jobs create \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --display-name=$DISPLAY_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URI \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
  --args="$CMDARGS"


# Run training with No GPU
# export EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest"
# gcloud ai custom-jobs create \
#   --project=$GCP_PROJECT \
#   --region=$GCP_REGION \
#   --display-name=$DISPLAY_NAME \
#   --python-package-uris=$PYTHON_PACKAGE_URI \
#   --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
#   --args="$CMDARGS"