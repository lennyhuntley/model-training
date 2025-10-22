# -*- coding: utf-8 -*-
"""One-time script to download NABirds from Deep Lake, convert to TFRecords, and upload to GCS."""

import os
import argparse
import tensorflow as tf
import deeplake as dl
from google.cloud import storage
import numpy as np

# --- Constants ---
HUB_TRAIN = "hub://activeloop/nabirds-dataset-train"
HUB_VAL = "hub://activeloop/nabirds-dataset-val"

# --- TFRecord Serialization Functions ---

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    """Creates a tf.train.Example message ready to be written to a file."""
    # Note: Images are stored as raw bytes, not encoded as JPEG/PNG.
    # This is faster to read in the training pipeline.
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False

def process_split(hub_path, local_filename, bucket_name, gcs_path):
    """Processes a single dataset split (train or val)."""
    print(f"--- Processing split: {hub_path} ---")
    ds = dl.load(hub_path, read_only=True)
    print(f"Loaded {len(ds)} samples from Deep Lake.")

    # Write to a local compressed TFRecord file
    with tf.io.TFRecordWriter(local_filename, options="GZIP") as writer:
        for i, sample in enumerate(ds):
            if i % 2000 == 0:
                print(f"  ... serialized {i}/{len(ds)} samples")
            image = sample.images.numpy()
            label = sample.labels.numpy(fetch_chunks=True).item()
            writer.write(serialize_example(image, label))
    
    print(f"Successfully created local TFRecord file: {local_filename}")

    # Upload to GCS
    print(f"Uploading {local_filename} to GCS bucket {bucket_name}...")
    upload_to_gcs(bucket_name, local_filename, gcs_path)

def main(args):
    os.makedirs(args.local_dir, exist_ok=True)

    # --- Process Training Set ---
    train_filename = os.path.join(args.local_dir, "nabirds_train.tfrecord.gz")
    train_gcs_path = "data/nabirds/nabirds_train.tfrecord.gz"
    process_split(HUB_TRAIN, train_filename, args.gcs_bucket, train_gcs_path)

    # --- Process Validation Set ---
    val_filename = os.path.join(args.local_dir, "nabirds_val.tfrecord.gz")
    val_gcs_path = "data/nabirds/nabirds_val.tfrecord.gz"
    process_split(HUB_VAL, val_filename, args.gcs_bucket, val_gcs_path)

    print("\n--- All splits processed and uploaded! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NABirds Deep Lake dataset to TFRecords and upload to GCS.")
    parser.add_argument("--gcs_bucket", type=str, required=True, help="GCS bucket name to upload TFRecords to.")
    parser.add_argument("--local_dir", type=str, default="./tfrecord_data", help="Local directory to temporarily store TFRecord files.")
    
    # Ensure deeplake is v3
    try:
        import deeplake as dl
        _v = tuple(int(x) for x in dl.__version__.split(".")[:2])
        if _v >= (4, 0):
            raise ImportError("deeplake v4+ detected. This script requires v3.")
    except Exception:
        print("Installing deeplake<4...")
        os.system("pip install 'deeplake<4'")

    parsed_args = parser.parse_args()
    main(parsed_args)
