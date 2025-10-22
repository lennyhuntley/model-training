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
from google.cloud import storage

# --- Argument Parsing & Configuration ---

def get_args():
    parser = argparse.ArgumentParser(description="Vertex AI Fine-tuning for NABirds")

    # --- W&B and GCS --- 
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key.")
    parser.add_argument("--gcs_data_dir", type=str, required=True, help="GCS directory containing the TFRecord data.")
    parser.add_argument("--gcs_bucket", type=str, default="know-now-app-training-data-lh", help="GCS bucket for saving the final model.")

    # --- Dataset --- 
    
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

# Dataset constants
NUM_CLASSES = 555
NUM_TRAIN_SAMPLES = 23928
NUM_VAL_SAMPLES = 24633

# --- TFRecord Parsing Functions ---
def _parse_tfrecord_fn(example):
    """Parses a single TFRecord example."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    image.set_shape([None, None, 3]) # Reshape from raw bytes
    label = tf.cast(example['label'], tf.int32)
    return image, label

def create_dataset_from_tfrecords(gcs_path, img_size, num_classes, training, batch_size):
    """Creates a tf.data.Dataset from TFRecords in GCS."""
    dataset = tf.data.Dataset.list_files(gcs_path, shuffle=training)

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training
    )

    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def _prep(img, lbl):
        img = tf.image.resize(img, [img_size, img_size], antialias=True)
        lbl = tf.one_hot(lbl, num_classes, dtype=tf.float32)
        return img, lbl

    dataset = dataset.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(10000) # Shuffle records

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets from TFRecords
train_path = os.path.join(args.gcs_data_dir, "nabirds_train.tfrecord.gz")
val_path = os.path.join(args.gcs_data_dir, "nabirds_val.tfrecord.gz")

train_ds = create_dataset_from_tfrecords(
    train_path,
    args.img_size,
    NUM_CLASSES,
    training=True,
    batch_size=args.batch_size
)
val_ds = create_dataset_from_tfrecords(
    val_path,
    args.img_size,
    NUM_CLASSES,
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
        layers.RandomRotation(0.05),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

def build_model(num_classes, img_size):
    """Builds the MobileNetV2 model for fine-tuning."""
    inputs = layers.Input(shape=(img_size, img_size, 3), dtype="uint8")
    
    # Cast to float32 for augmentation and preprocessing within a Keras layer
    x = layers.Lambda(lambda t: tf.cast(t, "float32"))(inputs)

    # Augmentation
    x = make_augmenter(img_size)(x)

    
    # Backbone
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    backbone.trainable = False  # Start with a frozen backbone

    # Classification Head
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    
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

class LogLearningRate(keras.callbacks.Callback):
    """Logs the current learning rate to W&B."""
    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
                # Get the value from the schedule at the current step
                step = self.model.optimizer.iterations
                lr_value = lr(step)
                wandb.log({'learning_rate': lr_value.numpy()}, commit=False)
            else:
                # For a static learning rate
                wandb.log({'learning_rate': keras.backend.get_value(lr)}, commit=False)
        except Exception as e:
            print(f"W&B: Could not log learning rate: {e}")

# --- Training --- 
print("\n--- Training Initializing ---")

# W&B Login
wdb_key = os.environ.get("WANDB_API_KEY") or args.wandb_key
if not wdb_key:
    raise ValueError("W&B API key not found. Please set --wandb_key or WANDB_API_KEY env var.")
wdb_project = "nabirds-finetuning-vertex"
wdb_run_name = f"{args.model_name}-{int(time.time())}"
wdb_config = vars(args)

wdb_config["num_classes"] = NUM_CLASSES

wandb.login(key=wdb_key)
wdb = wandb.init(project=wdb_project, name=wdb_run_name, config=wdb_config)

# Build the model inside the strategy scope
with strategy.scope():
    model = build_model(NUM_CLASSES, args.img_size)
    print(model.summary())

    # --- 1. Warmup Phase ---
    print("\n--- Phase 1: Warmup ---")
    model.compile(
        optimizer=Adam(learning_rate=args.lr_warmup),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

    steps_per_epoch = NUM_TRAIN_SAMPLES // args.batch_size
    validation_steps = int(math.ceil(NUM_VAL_SAMPLES / args.batch_size))

    model.fit(
        train_ds.repeat(),
        validation_data=val_ds.repeat(),
        epochs=args.epochs_warmup,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[wandb.keras.WandbMetricsLogger(log_freq="epoch")],
        verbose=2
    )

    # --- 2. Fine-tuning Phase ---
    print("\n--- Phase 2: Fine-tuning ---")
    unfreeze_top(model, ratio=args.freeze_ratio, freeze_bn=True)

    total_ft_steps = steps_per_epoch * args.epochs_finetune
    lr_schedule = CosineDecay(initial_learning_rate=args.lr_fine, decay_steps=total_ft_steps, alpha=0.1)

    with strategy.scope():
        model.compile(
            optimizer=AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay, clipnorm=1.0),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top5_accuracy")]
        )

    # Callbacks for fine-tuning
    # Note: W&B Model Checkpointing saves the model to W&B servers
    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq="epoch"),
        keras.callbacks.ModelCheckpoint(
            filepath=f"gs://{args.gcs_bucket}/checkpoints/{wdb_run_name}/best_model.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True, verbose=1
        ),
        LogLearningRate()
    ]

    history_ft = model.fit(
        train_ds.repeat(),
        validation_data=val_ds.repeat(),
        epochs=args.epochs_finetune,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2
    )

# --- Evaluation and Saving ---
print("\n--- Final Evaluation ---")
eval_out = model.evaluate(val_ds, steps=validation_steps, verbose=1)
final_metrics = dict(zip(model.metrics_names, eval_out))
print("Final eval metrics:", final_metrics)
wdb.log({"final_eval_" + k: v for k, v in final_metrics.items()})

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

# Finish W&B run
wdb.finish()

print("\n--- Training Job Complete ---")
