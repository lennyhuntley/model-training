import argparse
import os
import requests
import zipfile
import tarfile
import time


# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import count_params


# TensorFlow.js converter (imported only when needed)


# sklearn
from sklearn.model_selection import train_test_split



# W&B
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger



# Setup the arguments for the trainer task
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-dir", dest="model_dir", default="test", type=str, help="Model dir."
)
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--model_name",
    dest="model_name",
    default="mobilenetv2",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--train_base",
    dest="train_base",
    default=False,
    action="store_true",
    help="Train base or not",
)
parser.add_argument(
    "--epochs", dest="epochs", default=10, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key"
)

# Parse the arguments after all are defined
args = parser.parse_args()


# TF Version
print("tensorflow version", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())
# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))


print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("All Physical Devices", tf.config.list_physical_devices())



# Utils functions
def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)


    packet_file = os.path.basename(packet_url)


    # Handle GCS URLs
    if packet_url.startswith("gs://"):
        # Use tf.io.gfile for GCS (works on Vertex AI)
        local_path = os.path.join(base_path, packet_file) if base_path else packet_file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tf.io.gfile.copy(packet_url, local_path, overwrite=True)
    else:
        # Use requests for HTTP URLs
        with requests.get(packet_url, stream=True, headers=headers) as r:
            r.raise_for_status()
            local_path = os.path.join(base_path, packet_file) if base_path else packet_file
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(local_path) as zfile:
                zfile.extractall(base_path or ".")
        else:
            packet_name = packet_file.split(".")[0]
            with tarfile.open(local_path) as tfile:
                tfile.extractall(base_path or ".")



# Download Data
print("Downloading data from GCS...")
start_time = time.time()
download_file(
    "gs://know-now-app-training-data-lh/nabirds_mini_stratified.zip",
    base_path="datasets",
    extract=True,
)
execution_time = (time.time() - start_time) / 60.0
print("Download execution time (mins)", execution_time)


# Load Data
base_path = os.path.join("datasets", "nabirds_mini_stratified")
label_names = os.listdir(base_path)
print("Labels:", label_names)


# Number of unique labels
num_classes = len(label_names)
# Create label index for easy lookup
label2index = dict((name, index) for index, name in enumerate(label_names))
index2label = dict((index, name) for index, name in enumerate(label_names))


# Generate a list of labels and path to images
data_list = []
for label in label_names:
    class_dir = os.path.join(base_path, label)
    # Only process directories (skip any non-directory files)
    if os.path.isdir(class_dir):
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        data_list.extend([(label, os.path.join(class_dir, f)) for f in image_files])


print("Full size of the dataset:", len(data_list))
print("data_list:", data_list[:5])


# Load X & Y
# Build data x, y
data_x = [itm[1] for itm in data_list]
data_y = [itm[0] for itm in data_list]
print("data_x:", len(data_x))
print("data_y:", len(data_y))
print("data_x:", data_x[:5])
print("data_y:", data_y[:5])


# Split Data
test_percent = 0.10
validation_percent = 0.2


# Check class distribution before stratified split
from collections import Counter
class_counts = Counter(data_y)
print("Class distribution:", class_counts)


# Check if any class has fewer than 2 samples
classes_with_insufficient_data = [cls for cls, count in class_counts.items() if count < 2]
if classes_with_insufficient_data:
    print(f"Warning: Classes with < 2 samples: {classes_with_insufficient_data}")
    print("Removing classes with insufficient data for stratified split")
    # Filter out classes with < 2 samples
    sufficient_data_indices = [i for i, label in enumerate(data_y) if class_counts[label] >= 2]
    data_x = [data_x[i] for i in sufficient_data_indices]
    data_y = [data_y[i] for i in sufficient_data_indices]
    print(f"Filtered dataset size: {len(data_x)} samples")


    # Update label mappings
    label_names = sorted(set(data_y))
    num_classes = len(label_names)
    label2index = {name: idx for idx, name in enumerate(label_names)}
    index2label = {idx: name for idx, name in enumerate(label_names)}
    print("Updated labels:", label_names)


# Now perform stratified split safely
stratify_data_y = data_y if len(set(data_y)) > 1 and all(count >= 2 for count in Counter(data_y).values()) else None


# Split data into train / test
train_validate_x, test_x, train_validate_y, test_y = train_test_split(
    data_x, data_y, test_size=test_percent, stratify=stratify_data_y
)


# Check again for the train_validate split
train_validate_class_counts = Counter(train_validate_y)
stratify_train_validate_y = train_validate_y if len(set(train_validate_y)) > 1 and all(count >= 2 for count in train_validate_class_counts.values()) else None


# For small datasets, check if stratified split would create insufficient samples per class
min_samples_per_class = 2  # Minimum samples needed per class for reliable splitting
required_validation_samples = num_classes * min_samples_per_class
validation_samples = int(len(train_validate_x) * test_percent)


if stratify_train_validate_y is not None and validation_samples < required_validation_samples:
    print(f"Dataset too small for stratified validation split ({validation_samples} < {required_validation_samples}). Using random split.")
    stratify_train_validate_y = None  # Disable stratification


# Split data into train / validate
train_x, validate_x, train_y, validate_y = train_test_split(
    train_validate_x, train_validate_y, test_size=test_percent,
    stratify=stratify_train_validate_y, random_state=42
)


print("train_x count:", len(train_x))
print("validate_x count:", len(validate_x))
print("test_x count:", len(test_x))


# Login into wandb
wandb.login(key=args.wandb_key)



# Create TF Datasets
def get_dataset(image_width=224, image_height=224, num_channels=3, batch_size=32):
    # Load Image
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image, label


    # Normalize pixels
    def normalize(image, label):
        image = image / 255
        return image, label


    train_shuffle_buffer_size = len(train_x)
    validation_shuffle_buffer_size = len(validate_x)


    # Convert all y labels to numbers
    train_processed_y = [label2index[label] for label in train_y]
    validate_processed_y = [label2index[label] for label in validate_y]
    test_processed_y = [label2index[label] for label in test_y]


    # Converts to y to binary class matrix (One-hot-encoded)
    train_processed_y = to_categorical(train_processed_y, num_classes=num_classes)
    validate_processed_y = to_categorical(validate_processed_y, num_classes=num_classes)
    test_processed_y = to_categorical(test_processed_y, num_classes=num_classes)


    # Create TF Dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_processed_y))
    validation_data = tf.data.Dataset.from_tensor_slices(
        (validate_x, validate_processed_y)
    )
    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_processed_y))


    #############
    # Train data
    #############
    # Apply all data processing logic
    train_data = train_data.shuffle(buffer_size=train_shuffle_buffer_size)
    train_data = train_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)


    ##################
    # Validation data
    ##################
    # Apply all data processing logic
    validation_data = validation_data.shuffle(
        buffer_size=validation_shuffle_buffer_size
    )
    validation_data = validation_data.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_data = validation_data.map(
        normalize, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_data = validation_data.batch(batch_size)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)


    ############
    # Test data
    ############
    # Apply all data processing logic
    test_data = test_data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)


    return (train_data, validation_data, test_data)



def build_mobilenet_model(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels


    # Load a pretrained model from keras.applications
    tranfer_model_base = keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False
    )


    # Freeze the mobileNet model layers
    tranfer_model_base.trainable = train_base



    # Regularize using L1
    kernel_weight = 0.02
    bias_weight = 0.02


    model = Sequential(
        [
            tranfer_model_base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
            keras.layers.Dense(
                units=num_classes,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
        ],
        name=model_name + "_train_base_" + str(train_base),
    )


    return model



print("Train model")
############################
# Training Params
############################
model_name = args.model_name
learning_rate = 0.001
image_width = 224
image_height = 224
num_channels = 3
batch_size = args.batch_size
epochs = args.epochs
train_base = args.train_base


# Free up memory
K.clear_session()


# Data
train_data, validation_data, test_data = get_dataset(
    image_width=image_width,
    image_height=image_height,
    num_channels=num_channels,
    batch_size=batch_size,
)



# Model
model = build_mobilenet_model(
    image_height,
    image_width,
    num_channels,
    num_classes,
    model_name,
    train_base=train_base,
)
# Optimizer
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
# Loss
loss = keras.losses.categorical_crossentropy
# Print the model architecture
print(model.summary())
# Compile
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])


# Initialize a W&B run
wandb.init(
    project="nabirds-mini-training",  # (optional) renamed from 'cheese' to reflect dataset
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": model.name,
    },
    name=model.name,
)


# Train model
start_time = time.time()
training_results = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    verbose=1,
)
execution_time = (time.time() - start_time) / 60.0
print("Training execution time (mins)", execution_time)


# Save the trained model to the file system
model_save_path = "trained_model"
print(f"Saving model to: {model_save_path}")
model.save(model_save_path)
print(f"✅ Model saved successfully to: {model_save_path}")

# Upload model to GCS
gcs_model_path = f"gs://know-now-app-training-data-lh/models/{model.name}_{int(time.time())}"
print(f"Uploading model to GCS: {gcs_model_path}")

try:
    # Create a tar.gz of the model directory for upload
    import tarfile
    tar_path = f"{model_save_path}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_save_path, arcname=os.path.basename(model_save_path))

    # Upload the tar.gz file
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("know-now-app-training-data-lh")
    blob = bucket.blob(os.path.basename(gcs_model_path) + '.tar.gz')
    blob.upload_from_filename(tar_path)
    print(f"Model uploaded successfully to: {gcs_model_path}.tar.gz")

    # Clean up local tar file
    os.remove(tar_path)


except ImportError:
    try:
        # Fall back to using tf.io.gfile for basic copy
        tf.io.gfile.copy(model_save_path, gcs_model_path, overwrite=True)
        print(f"Model uploaded successfully to: {gcs_model_path}")
    except Exception as e:
        print(f"Could not upload model to GCS: {e}")
        print("Model is saved locally and can be manually copied to GCS")
except Exception as e:
    print(f"Error during model upload: {e}")
    print("Model is saved locally and can be manually copied to GCS")


# Update W&B
wandb.config.update({"execution_time": execution_time})
# Close the W&B run
wandb.run.finish()



print("Training Job Complete")
