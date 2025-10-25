#!/bin/sh

set -eu

if [ "$#" -lt 1 ]; then
  cat <<'EOF'
Usage: convert_model_to_tfjs.sh gs://bucket/path/to/model.keras [mobile_subdir]

Arguments:
  gs://bucket/path/to/model.keras  Full GCS path to the exported `.keras` model file.
  mobile_subdir                    Optional subdirectory name under `mobile/` for upload.
                                   Defaults to the parent directory of the model file.
EOF
  exit 1
fi

SOURCE_MODEL_PATH="$1"
MOBILE_SUBDIR="${2:-$(basename "$(dirname "$SOURCE_MODEL_PATH")")}"

SOURCE_GCS_PATH="${SOURCE_MODEL_PATH#gs://}"
SOURCE_BUCKET="${SOURCE_GCS_PATH%%/*}"
DEFAULT_BUCKET_URI="gs://${SOURCE_BUCKET}"
TARGET_BUCKET_URI="${MOBILE_BUCKET_URI:-$DEFAULT_BUCKET_URI}"

if [ "${TARGET_BUCKET_URI}" != "${DEFAULT_BUCKET_URI}" ]; then
  echo "Resolved upload bucket override: $TARGET_BUCKET_URI (default would be $DEFAULT_BUCKET_URI)"
else
  echo "Resolved upload bucket: $TARGET_BUCKET_URI"
fi

case "$SOURCE_MODEL_PATH" in
  gs://*) ;;
  *)
    echo "ERROR: Source model path must begin with gs://" >&2
    exit 1
    ;;
esac

if ! command -v gsutil >/dev/null 2>&1; then
  echo "ERROR: gsutil command not found. Ensure Google Cloud SDK is installed in the docker shell." >&2
  exit 1
fi

if ! python -c "import tensorflowjs" >/dev/null 2>&1; then
  echo "ERROR: tensorflowjs Python package not available. Rebuild the docker image so dependencies are installed." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

LOCAL_MODEL_PATH="$TMP_DIR/model.keras"
LOCAL_SAVED_MODEL_DIR="$TMP_DIR/saved_model"
LOCAL_INFERENCE_MODEL_PATH="$TMP_DIR/model_inference.h5"
OUTPUT_DIR="$TMP_DIR/tfjs_model"
CLASS_DATA_DIR="${CLASS_DATA_DIR:-gs://${SOURCE_BUCKET}/nabirds_preprocessed}"
CLASS_MAP_PATH="${CLASS_DATA_DIR%/}/class_map.json"
CLASS_NAMES_PATH="${CLASS_DATA_DIR%/}/classes.txt"
LABEL_CATEGORY_NAME="${LABEL_CATEGORY_NAME:-Category}"
LABELS_JSON_PATH="$OUTPUT_DIR/labels.json"
LOCAL_ASSETS_DIR="${LOCAL_ASSETS_DIR:-assets/models/birds}"

echo "Downloading $SOURCE_MODEL_PATH -> $LOCAL_MODEL_PATH"
gsutil cp "$SOURCE_MODEL_PATH" "$LOCAL_MODEL_PATH"

echo "Exporting .keras archive to SavedModel..."
python - "$LOCAL_MODEL_PATH" "$LOCAL_SAVED_MODEL_DIR" "$LOCAL_INFERENCE_MODEL_PATH" "$LABELS_JSON_PATH" "$CLASS_MAP_PATH" "$CLASS_NAMES_PATH" "$OUTPUT_DIR" "$LABEL_CATEGORY_NAME" <<'PY'
import sys
from pathlib import Path
import tensorflow as tf
import keras
import json
import h5py
import math

from tensorflowjs.converters import save_keras_model

try:
    from keras.src.saving import serialization_lib
except ImportError:  # pragma: no cover - fallback for older Keras releases
    from keras.saving import serialization_lib

keras_path = Path(sys.argv[1])
saved_model_dir = Path(sys.argv[2])
inference_keras_path = Path(sys.argv[3])
labels_output = Path(sys.argv[4])
class_map_uri = sys.argv[5]
classes_txt_uri = sys.argv[6]
output_dir = Path(sys.argv[7])
category_label = sys.argv[8]

class CompatibleRandomRotation(keras.layers.RandomRotation):
    def __init__(self, *args, value_range=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        cfg.pop("value_range", None)
        return super().from_config(cfg)

class CompatibleLambda(keras.layers.Lambda):
    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        if cfg.get("output_shape") is None:
            cfg["output_shape"] = lambda shape: shape
        layer = super().from_config(cfg)
        layer.__class__ = cls
        return layer

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, input_spec, *args, **kwargs):
        try:
            return super().compute_output_spec(input_spec, *args, **kwargs)
        except (NotImplementedError, TypeError):
            return input_spec

keras.layers.Lambda = CompatibleLambda

try:
    from keras.layers.core import lambda_layer as core_lambda_module
except ImportError:
    core_lambda_module = None

if core_lambda_module is not None:
    core_lambda_module.Lambda = CompatibleLambda

try:
    from keras.src.layers.core import lambda_layer as src_lambda_module
except ImportError:
    src_lambda_module = None

if src_lambda_module is not None:
    src_lambda_module.Lambda = CompatibleLambda


def _ensure_lambda_methods(lambda_cls):
    compatible_shape = CompatibleLambda.compute_output_shape
    compatible_spec = CompatibleLambda.compute_output_spec

    if getattr(lambda_cls.compute_output_shape, "__func__", lambda_cls.compute_output_shape) is not compatible_shape:
        lambda_cls.compute_output_shape = compatible_shape

    if getattr(lambda_cls.compute_output_spec, "__func__", lambda_cls.compute_output_spec) is not compatible_spec:
        lambda_cls.compute_output_spec = compatible_spec


for _lambda_module in filter(None, [keras.layers, core_lambda_module, src_lambda_module]):
    lambda_cls = getattr(_lambda_module, "Lambda", None)
    if lambda_cls is not None:
        _ensure_lambda_methods(lambda_cls)

custom_objects_map = {
    "Lambda": CompatibleLambda,
    "keras.layers.Lambda": CompatibleLambda,
    "keras.src.layers.core.lambda_layer.Lambda": CompatibleLambda,
    "tf.keras.layers.Lambda": CompatibleLambda,
    "__lambda__": CompatibleLambda,
}

keras.utils.get_custom_objects().update(custom_objects_map)
tf.keras.utils.get_custom_objects().update(custom_objects_map)

custom_objects = {
    "RandomRotation": CompatibleRandomRotation,
    "keras.layers.RandomRotation": CompatibleRandomRotation,
    "keras.src.layers.preprocessing.image_preprocessing.random_rotation.RandomRotation": CompatibleRandomRotation,
    **custom_objects_map,
}

model = tf.keras.models.load_model(
    keras_path,
    compile=False,
    safe_mode=False,
    custom_objects=custom_objects,
)

def _is_random_layer(layer):
    class_name = layer.__class__.__name__
    module_name = layer.__class__.__module__
    name = getattr(layer, "name", "")
    if class_name.startswith("Random"):
        return True
    if "random" in name.lower():
        return True
    if "random" in module_name:
        return True
    return False

def _identity_layer(original_layer):
    return keras.layers.Lambda(lambda x: x, name=f"{original_layer.name}_identity")

def _clone_inference_model(model):
    def clone_fn(layer):
        if isinstance(layer, keras.Sequential) and "augmentation" in layer.name.lower():
            return _identity_layer(layer)
        if _is_random_layer(layer):
            return _identity_layer(layer)
        return layer

    cloned = keras.models.clone_model(model, clone_function=clone_fn)
    cloned.set_weights(model.get_weights())
    return cloned

model = _clone_inference_model(model)


def _has_target_rescaling(layer):
    if not isinstance(layer, tf.keras.layers.Rescaling):
        return False
    try:
        scale = float(layer.scale)
        offset = float(layer.offset)
    except (TypeError, ValueError):
        return False
    return math.isclose(scale, 1.0 / 127.5, rel_tol=1e-6) and math.isclose(offset, -1.0, rel_tol=1e-6)


def _ensure_rescaling_layer(model):
    for layer in model.layers:
        if _has_target_rescaling(layer):
            return model
    inputs = tf.keras.Input(shape=model.inputs[0].shape[1:], dtype=tf.float32, name="pixel_input")
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="auto_rescaling")(inputs)
    outputs = model(x, training=False)
    wrapped = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_with_rescale")
    return wrapped


def _ensure_float32_outputs(model):
    outputs = []
    changed = False
    for tensor in model.outputs:
        if tensor.dtype != tf.float32:
            outputs.append(tf.cast(tensor, tf.float32))
            changed = True
        else:
            outputs.append(tensor)
    if not changed:
        return model
    wrapped = tf.keras.Model(inputs=model.inputs, outputs=outputs, name=f"{model.name}_float32")
    return wrapped


model = _ensure_rescaling_layer(model)
model = _ensure_float32_outputs(model)

def _load_raw_to_idx(uri):
    try:
        with tf.io.gfile.GFile(uri, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: could not load class map from {uri}: {exc}")
        return {}
    mapping = data.get("raw_to_idx", data)
    out = {}
    for raw, idx in mapping.items():
        try:
            raw_val = int(raw)
            idx_val = int(idx)
        except Exception:
            continue
        out[raw_val] = idx_val
    return out

def _load_class_names(uri):
    names = {}
    try:
        with tf.io.gfile.GFile(uri, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as exc:
        print(f"Warning: could not load class names from {uri}: {exc}")
        return names
    for line in lines:
        raw_str = None
        name = None
        if "," in line:
            raw_str, name = line.split(",", 1)
        elif "\t" in line:
            raw_str, name = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                raw_str, name = parts
        if raw_str is None or name is None:
            continue
        try:
            raw_id = int(raw_str.strip())
        except ValueError:
            continue
        cleaned_name = name.strip()
        if not cleaned_name:
            continue
        names[raw_id] = cleaned_name
    return names

def _save_labels(output_path, raw_to_idx, class_names, category_title):
    if not raw_to_idx:
        print("Warning: raw_to_idx mapping empty; skipping labels.json generation")
        return
    max_idx = max(raw_to_idx.values())
    names = [None] * (max_idx + 1)
    for raw_id, idx in raw_to_idx.items():
        if idx < 0:
            continue
        label = class_names.get(raw_id, str(raw_id))
        if idx >= len(names):
            names.extend([None] * (idx + 1 - len(names)))
        names[idx] = label
    for i, value in enumerate(names):
        if value is None:
            names[i] = str(i)
    payload = [category_title] + names
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved labels metadata to {output_path}")

raw_to_idx_map = _load_raw_to_idx(class_map_uri)
class_name_map = _load_class_names(classes_txt_uri)

if saved_model_dir.exists():
    import shutil
    shutil.rmtree(saved_model_dir)

try:
    model.export(saved_model_dir)
except AttributeError:
    tf.saved_model.save(model, saved_model_dir)

inference_keras_path.parent.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
try:
    model.save(inference_keras_path, include_optimizer=False, save_format="h5")
except TypeError:
    tf.keras.models.save_model(model, inference_keras_path, include_optimizer=False, save_format="h5")

with h5py.File(inference_keras_path, "r"):
    pass

_save_labels(labels_output, raw_to_idx_map, class_name_map, category_label)

save_keras_model(model, str(output_dir))
PY

echo "Syncing TensorFlow.js artifacts to local assets directory: $LOCAL_ASSETS_DIR"
mkdir -p "$(dirname "$LOCAL_ASSETS_DIR")"
rm -rf "$LOCAL_ASSETS_DIR"
mkdir -p "$LOCAL_ASSETS_DIR"
cp -a "$OUTPUT_DIR/." "$LOCAL_ASSETS_DIR/"

TARGET_URI="${TARGET_BUCKET_URI%/}/mobile/${MOBILE_SUBDIR}"

echo "Uploading TensorFlow.js artifacts to $TARGET_URI"
gsutil -m rsync -r "$OUTPUT_DIR" "$TARGET_URI"

echo "✅ Conversion complete. Artifacts available at $TARGET_URI"
