#!/bin/bash

set -e

# This script is designed to be run from within the Docker container
# started by the docker-shell.sh script.

echo "Starting data upload to GCS..."

gsutil -m cp -r /app/logs/kaggle_nabirds/nabirds_preprocessed gs://kaggle_nabirds_data/

echo "Data upload complete."
