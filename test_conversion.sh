#!/bin/bash
# TensorFlow.js Conversion Test Script
# Run this after downloading a model from GCS

set -e  # Exit on any error

echo "🧪 Starting TensorFlow.js Conversion Test"
echo "========================================"

# Check if model exists
if [ ! -d "trained_model" ]; then
    echo "❌ Error: trained_model directory not found"
    echo "💡 Make sure you've extracted a model from GCS first"
    exit 1
fi

echo "✅ Found trained_model directory"

# Install tensorflowjs if not already installed
echo "📦 Installing TensorFlow.js converter..."
pip install tensorflowjs==4.15.0

# Convert the model
echo "🔄 Converting SavedModel to TensorFlow.js..."
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  trained_model/ \
  model_tfjs/

echo "✅ Conversion completed!"

# Verify the output
echo "🔍 Checking converted files..."
ls -la model_tfjs/

if [ -f "model_tfjs/model.json" ] && [ -f "model_tfjs/group1-shard1of1.bin" ]; then
    echo "✅ TensorFlow.js model files created successfully"

    # Quick validation with Python
    echo "🧪 Running quick validation..."
    python3 -c "
import tensorflowjs as tfjs
import json

# Load the converted model
model = tfjs.converters.load_keras_model('model_tfjs/model.json')
print('✅ Model loaded successfully!')
print('Input shape:', model.input.shape)
print('Output shape:', model.output.shape)

# Check if labels exist
try:
    with open('model_tfjs/labels.json', 'r') as f:
        labels = json.load(f)
    print('✅ Labels found:', len(labels), 'classes')
except FileNotFoundError:
    print('⚠️  No labels.json found - you may need to create this manually')
"

    echo ""
    echo "🎉 Conversion test PASSED!"
    echo ""
    echo "📁 Your TensorFlow.js model is ready at: ./model_tfjs/"
    echo "📋 Next steps:"
    echo "   1. Upload model_tfjs/ to your web hosting"
    echo "   2. Use this URL in your React Native app:"
    echo "      const model = await tf.loadGraphModel('YOUR_HOSTING_URL/model.json');"

else
    echo "❌ Conversion test FAILED - missing required files"
    exit 1
fi
