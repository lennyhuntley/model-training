# -*- coding: utf-8 -*-
"""Vertex AI fine-tuning script for NABirds using MobileNetV2."""

import os
import argparse
import time
import math
import sys
import numpy as np
import shutil

# Set Keras backend before importing
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# --- Main Imports ---
import tensorflow as tf
import keras
from keras import layers, models
from keras import mixed_precision
from keras.optimizers import Adam, AdamW
from keras.optimizers.schedules import CosineDecay
from keras.metrics import TopKCategoricalAccuracy

# --- Library Imports ---
import deeplake as dl
import wandb
from wandb.keras import WandbMetricsLogger
from google.cloud import storage

# --- Argument Parsing & Configuration ---

def get_args():
    parser = argparse.ArgumentParser(description="Vertex AI Fine-tuning for NABirds")

    # --- W&B and GCS --- 
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key.")
    parser.add_argument("--gcs_data_dir", type=str, default="gs://kaggle_nabirds_data/nabirds_preprocessed", help="GCS directory containing the NABirds dataset.")
    parser.add_argument("--gcs_bucket", type=str, default="kaggle_nabirds_data", help="GCS bucket for saving the final model.")

    # --- Dataset --- 
    
    # --- Dataset ---
    parser.add_argument("--percent_to_use", type=float, default=1.0, help="Percentage of the training dataset to use (0.0 to 1.0).")

    # --- Model --- 
    parser.add_argument("--model_name", type=str, default="mobilenetv2_nabirds_finetuned", help="Base name for the trained model.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (height and width).")

    # --- Training Phases ---
    parser.add_argument("--epochs_warmup", type=int, default=2, help="Epochs for the warmup phase (frozen backbone).")
    parser.add_argument("--epochs_finetune", type=int, default=2, help="Epochs for the fine-tuning phase.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation.")

    # --- Hyperparameters ---
    parser.add_argument("--lr_warmup", type=float, default=1e-3, help="Learning rate for the warmup phase.")
    parser.add_argument("--lr_fine", type=float, default=3e-4, help="Base learning rate for the fine-tuning phase.")
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay for AdamW optimizer in fine-tuning.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for the loss function.")
    parser.add_argument("--freeze_ratio", type=float, default=0.75, help="Ratio of backbone layers to keep frozen during fine-tuning (e.g., 0.75 unfreezes top 25%%).")

    return parser.parse_args()

# --- Initial Setup ---
args = get_args()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- GPU & Mixed Precision Setup ---
print("--- System Setup ---")
print("TF:", tf.__version__)
print("Keras:", keras.__version__)
print("Keras backend:", keras.backend.backend())

if tf.config.list_physical_devices('GPU'):
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
    # Allow gradual memory growth
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print(f"Could not set memory growth for {g}: {e}")
else:
    print("No GPU detected. Running on CPU.")
# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))



# --- Data Pipeline ---
print("\n--- Data Pipeline Setup ---")


# --- Image Dataset Functions ---

def load_image_and_label(image_path, label, bbox, img_size):
    """Loads an image from a path and prepares the label."""
    # Load the image from GCS
    image = tf.io.read_file(image_path)
    # Decode the image to a dense tensor
    image = tf.io.decode_jpeg(image, channels=3)
    shape = tf.shape(image)
    
    # Bbox is [x, y, width, height] in absolute pixel values.
    # tf.image.crop_to_bounding_box wants [offset_height, offset_width, target_height, target_width].
    offset_y = tf.cast(bbox[1], tf.int32)
    offset_x = tf.cast(bbox[0], tf.int32)
    target_height = tf.cast(bbox[3], tf.int32)
    target_width = tf.cast(bbox[2], tf.int32)

    # Crop the image
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
    
    # Resize the cropped image
    image = tf.image.resize(image, [img_size, img_size], antialias=True)
    return image, label

def create_dataset_from_gcs(gcs_dir, img_size, training, batch_size, percent_to_use=1.0):
    """Creates a tf.data.Dataset from image folders in GCS."""
    """Creates a tf.data.Dataset from image folders in GCS."""
    # Read the annotations CSV from GCS
    annotations_path = os.path.join(gcs_dir, 'annotations_all.csv')
    split_path = os.path.join(gcs_dir, 'train_test_split.txt')

    # Use tf.io.gfile to read files from GCS
    with tf.io.gfile.GFile(annotations_path, 'r') as f:
        # Skip header
        annotations = f.read().strip().split('\n')[1:]
    with tf.io.gfile.GFile(split_path, 'r') as f:
        splits = f.read().strip().split('\n')

    image_paths = []
    labels = []
    bboxes = []

    # Create a mapping from image_id to split (0 for train, 1 for test)
    split_map = {line.split(' ')[0]: int(line.split(' ')[1]) for line in splits}

    for line in annotations:
        parts = line.split(',')
        image_id = parts[0]
        class_id = int(parts[1])
        image_name = parts[2]

        # Determine if the image is in the desired split (train/test)
        is_test_image = split_map.get(image_id, -1)

        if (training and is_test_image == 0) or (not training and is_test_image == 1):
            # Construct the full GCS path to the image
            full_path = os.path.join(gcs_dir, 'images', image_name)
            image_paths.append(full_path)
            labels.append(class_id)
            # Extract bbox [x, y, width, height]
            bbox = [float(p) for p in parts[3:]]
            bboxes.append(bbox)

    # If using a subset for training, perform stratified sampling.
    if training and percent_to_use < 1.0:
        print(f"Performing stratified sampling to use {percent_to_use * 100:.0f}% of the training data.")
        # Group paths by label
        paths_by_label = {}
        for path, label, bbox in zip(image_paths, labels, bboxes):
            if label not in paths_by_label:
                paths_by_label[label] = []
            paths_by_label[label].append((path, bbox))

        # Sample from each group
        stratified_paths = []
        stratified_labels = []
        stratified_bboxes = []
        for label, path_bbox_pairs in paths_by_label.items():
            np.random.shuffle(path_bbox_pairs)
            num_to_take = max(1, int(len(path_bbox_pairs) * percent_to_use))
            for i in range(num_to_take):
                path, bbox = path_bbox_pairs[i]
                stratified_paths.append(path)
                stratified_labels.append(label)
                stratified_bboxes.append(bbox)
        
        image_paths = stratified_paths
        labels = stratified_labels
        bboxes = stratified_bboxes

        # Final shuffle of the stratified subset
        temp_dataset = list(zip(image_paths, labels, bboxes))
        np.random.shuffle(temp_dataset)
        image_paths, labels, bboxes = zip(*temp_dataset)
        print(f"Using {len(image_paths)} training samples after stratification.")

    # Convert to TensorFlow constants for robust dataset creation
    image_paths_tf = tf.constant(image_paths, dtype=tf.string)
    labels_tf = tf.constant(labels, dtype=tf.int32)
    bboxes_tf = tf.constant(bboxes, dtype=tf.float32)

    # Create a dataset from the final paths, labels, and bounding boxes
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tf, labels_tf, bboxes_tf))

    if training:
        dataset = dataset.shuffle(10000) # Shuffle records

    # Map the loading function
    # Determine number of classes from the labels
    num_classes = len(set(labels))
    print(f"Found {num_classes} classes.")

    # Map the loading and one-hot encoding function
    def process_path(path, lbl, bbox):
        img, lbl = load_image_and_label(path, lbl, bbox, img_size)
        return img, tf.one_hot(lbl, num_classes, dtype=tf.float32)

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)


    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, len(image_paths), num_classes

# Create datasets from GCS image folders

# The GCS directory should point to the 'nabirds_preprocessed' folder
gcs_preprocessed_dir = args.gcs_data_dir

train_ds, NUM_TRAIN_SAMPLES, NUM_CLASSES = create_dataset_from_gcs(
    gcs_preprocessed_dir,
    args.img_size,
    training=True,
    batch_size=args.batch_size,
    percent_to_use=args.percent_to_use
)

val_ds, NUM_VAL_SAMPLES, _ = create_dataset_from_gcs(
    gcs_preprocessed_dir,
    args.img_size,
    training=False,
    batch_size=args.batch_size
)

print("Data pipelines ready.")

# --- Model Building ---
print("\n--- Model Definition ---")

def make_augmenter(img_size):
    """Creates a Keras Sequential model for lightweight augmentation."""
    # For fine-grained tasks, heavy augmentation can sometimes hurt.
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

def build_model(num_classes, img_size):
    """Builds the MobileNetV2 model for fine-tuning."""
    inputs = layers.Input(shape=(img_size, img_size, 3), dtype="uint8")
    
    # Cast to float32 for augmentation and preprocessing within a Keras layer
    x = layers.Lambda(lambda t: tf.cast(t, "float32"))(inputs)

    # Augmentation
    x = make_augmenter(img_size)(x)

    # Pre-processing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    
    # Backbone
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    backbone.trainable = False  # Start with a frozen backbone

    # Pass the backbone's output to the classification head
    x = backbone(x) # Let the fine-tuning fit call control the training mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    
    # Use float32 for the final layer for numerical stability
    logits = layers.Dense(num_classes, dtype="float32", name="logits")(x)
    probs = layers.Activation("softmax", dtype="float32", name="probs")(logits)

    model = keras.Model(inputs, probs, name=args.model_name)
    model._backbone = backbone  # Attach for easy access later
    return model

def unfreeze_top(model: keras.Model, ratio: float, freeze_bn: bool = True):
    """Unfreezes the top `ratio` of layers in the model's backbone."""
    backbone = getattr(model, "_backbone", None)
    if backbone is None:
        print("WARNING: Model has no '_backbone' attribute to unfreeze.")
        return

    n_layers = len(backbone.layers)
    n_to_unfreeze = int(n_layers * (1 - ratio))
    
    print(f"Unfreezing top {n_to_unfreeze} / {n_layers} layers of the backbone.")

    for i, layer in enumerate(reversed(backbone.layers)):
        if i < n_to_unfreeze:
            layer.trainable = True
        else:
            layer.trainable = False
        
        # Optionally keep BatchNormalization layers frozen
        if freeze_bn and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

print("Model building functions defined.")


# --- Custom Callback for Two-Phase Training ---

class FinetuningCallback(keras.callbacks.Callback):
    """Callback to handle the transition from warmup to fine-tuning."""
    def __init__(self, warmup_epochs, finetune_lr_schedule, finetune_wd, freeze_ratio):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.finetune_lr_schedule = finetune_lr_schedule
        self.finetune_wd = finetune_wd
        self.freeze_ratio = freeze_ratio

    def on_epoch_begin(self, epoch, logs=None):
        # Check if it's the first epoch of the fine-tuning phase
        if epoch == self.warmup_epochs:
            print("\n--- Starting Fine-tuning Phase ---")
            # Unfreeze the top layers of the backbone
            unfreeze_top(self.model, ratio=self.freeze_ratio, freeze_bn=True)
            
            # Update the learning rate of the existing optimizer
            self.model.optimizer.learning_rate = self.finetune_lr_schedule
            print(f"Model layers unfrozen. Learning rate updated.")


# --- Training --- 
print("\n--- Training Initializing ---")

# W&B Login
wdb_key = os.environ.get("WANDB_API_KEY") or args.wandb_key
if not wdb_key:
    raise ValueError("W&B API key not found. Please set --wandb_key or WANDB_API_KEY env var.")
wdb_project = "mobilenetv2_kaggle"
wdb_run_name = f"{args.model_name}-{int(time.time())}"
wdb_config = vars(args)

wdb_config["num_classes"] = NUM_CLASSES

wandb.login(key=wdb_key)
wdb = wandb.init(project=wdb_project, name=wdb_run_name, config=wdb_config)

# Build the model inside the strategy scope
with strategy.scope():
    model = build_model(NUM_CLASSES, args.img_size)
    print(model.summary())

    # --- Unified Training Loop ---
    print("\n--- Unified Training Loop ---")
    
    # Initial compilation with the AdamW optimizer for the entire run
    model.compile(
        optimizer=AdamW(learning_rate=args.lr_warmup, weight_decay=args.weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"]
    )

    # Prepare variables for the callback
    steps_per_epoch = NUM_TRAIN_SAMPLES // args.batch_size
    validation_steps = int(math.ceil(NUM_VAL_SAMPLES / args.batch_size))
    total_epochs = args.epochs_warmup + args.epochs_finetune

    # Create the learning rate scheduler for the fine-tuning phase
    decay_steps = steps_per_epoch * args.epochs_finetune
    lr_schedule = CosineDecay(
        initial_learning_rate=args.lr_fine,
        decay_steps=decay_steps,
        alpha=0.1
    )

    # All callbacks for the entire run
    all_callbacks = [
        WandbMetricsLogger(),
        FinetuningCallback(
            warmup_epochs=args.epochs_warmup, 
            finetune_lr_schedule=lr_schedule, 
            finetune_wd=args.weight_decay,
            freeze_ratio=args.freeze_ratio
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"gs://{args.gcs_bucket}/checkpoints/{wdb_run_name}/best_model.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", 
            mode="max", 
            patience=5,  # Increased patience
            restore_best_weights=True, 
            verbose=1
        )
    ]

    # Single, unified model.fit call
    model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=all_callbacks,
        verbose=2
    )

    # --- Evaluation and Saving (inside strategy scope) ---
    print("\n--- Final Evaluation ---")
    # Re-compile the model to ensure metrics are correctly initialized for evaluation
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"]
    )
    eval_out = model.evaluate(val_ds, steps=validation_steps, verbose=1, return_dict=True)
    print("Final eval metrics:", eval_out)
    wdb.log({"final_eval_" + k: v for k, v in eval_out.items()})

    # Save the final model to a .keras file
    model_filename = f"{args.model_name}.keras"
    model.save(model_filename)
    print(f"Model saved locally to {model_filename}")

    # Upload to GCS
    print(f"\n--- Uploading to GCS ---")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(args.gcs_bucket)
        gcs_model_path = f"models/{wdb_run_name}/{model_filename}"
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(model_filename)
        print(f"✅ Model successfully uploaded to gs://{args.gcs_bucket}/{gcs_model_path}")
    except Exception as e:
        print(f"ERROR: Failed to upload model to GCS: {e}")

# Finish W&B run (outside scope)
wdb.finish()

print("\n--- Training Job Complete ---")
