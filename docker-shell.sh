#!/bin/bash

set -e

export IMAGE_NAME="model-training-cli"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_URI="gs://know-now-app-trainer-lh"
export GCP_PROJECT="ac215-475412"
export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"


# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME --platform=$DOCKER_PLATFORM -f Dockerfile .

# Run Container
docker run --rm --platform=$DOCKER_PLATFORM --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_URI=$GCS_BUCKET_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME