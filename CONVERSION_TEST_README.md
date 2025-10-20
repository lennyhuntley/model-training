# 🧪 TensorFlow.js Model Conversion Test Suite

This directory contains scripts to test whether your TensorFlow.js model conversion works correctly before implementing automation.

## 🎯 Purpose

Test the complete pipeline from SavedModel → TensorFlow.js → React Native compatibility **before** building automation.

## 📋 Test Workflow

### **Phase 1: Manual Conversion Test**
```bash
# 1. Download a model from your GCS bucket
gsutil cp gs://know-now-app-training-data-lh/models/YOUR_MODEL.tar.gz ./model.tar.gz

# 2. Extract the model
tar -xzf model.tar.gz

# 3. Run the conversion test
chmod +x test_conversion.sh
./test_conversion.sh
```

### **Phase 2: JavaScript Environment Test**
```bash
# 1. Install Node.js dependencies
npm install

# 2. Run the JavaScript model test
npm test
```

## 📁 File Structure

```
model-training/
├── test_conversion.sh    # Tests Python conversion process
├── test_model.js         # Tests model in JavaScript environment
├── package.json          # Node.js dependencies
└── trained_model/        # Extracted SavedModel (after download)
    └── model_tfjs/       # Converted TensorFlow.js model (after test)
```

## ✅ What Gets Tested

### **Conversion Test (`test_conversion.sh`)**
- ✅ Model file extraction from tar.gz
- ✅ TensorFlow.js converter installation
- ✅ SavedModel → TensorFlow.js conversion
- ✅ Output file verification
- ✅ Basic Python model loading test

### **JavaScript Test (`test_model.js`)**
- ✅ Model loading in JavaScript environment
- ✅ Input/output tensor shape validation
- ✅ Prediction generation capability
- ✅ Label file verification (if present)
- ✅ Memory cleanup verification

## 🎯 Expected Results

### **Successful Conversion Test Output:**
```
🧪 Starting TensorFlow.js Conversion Test
================================================
✅ Found trained_model directory
📦 Installing TensorFlow.js converter...
🔄 Converting SavedModel to TensorFlow.js...
✅ Conversion completed!
🔍 Checking converted files...
✅ TensorFlow.js model files created successfully
🧪 Running quick validation...
✅ Model loaded successfully!
Input shape: [null, 224, 224, 3]
Output shape: [null, 555]
✅ Labels found: 555 classes

🎉 Conversion test PASSED!
```

### **Successful JavaScript Test Output:**
```
🧪 Testing TensorFlow.js Model in Node.js Environment
=======================================================
📥 Loading model...
✅ Model loaded successfully!

📊 Model Information:
Input shape: [null, 224, 224, 3]
Output shape: [null, 555]

🧪 Creating test input...
Input tensor shape: [1, 224, 224, 3]
Input range: -2.34 to 2.87

🔮 Making prediction...
✅ Prediction completed!
Output shape: [1, 555]
Output range: 0.00 to 0.98

🎯 Top prediction index: 123
🎯 Confidence: 87.34%
🏷️  Predicted class: American Robin
📋 Available classes: 555 total

🎉 MODEL TEST PASSED!
```

## 🚨 Common Issues & Solutions

### **"No module named 'tensorflowjs'"**
```bash
pip install tensorflowjs==4.15.0
```

### **"Conversion failed"**
- Check that `trained_model/` contains `saved_model.pb`
- Verify the model was saved correctly during training

### **"Model loading failed in JavaScript"**
```bash
npm install  # Make sure dependencies are installed
# Check that model_tfjs/model.json exists
```

### **"Wrong input/output shapes"**
- Input should be `[null, 224, 224, 3]` for MobileNetV2
- Output should match your number of bird classes

## 📊 Test Results Interpretation

| Test | Status | Meaning |
|------|--------|---------|
| ✅ Conversion Test | **PASS** | Python conversion works correctly |
| ✅ JavaScript Test | **PASS** | Model works in JS environment |
| ✅ Prediction Shapes | **Correct** | Model architecture is valid |
| ✅ Labels Match | **Present** | Classification labels are available |

## 🎯 Next Steps After Successful Tests

1. **✅ Confirmed conversion works** → Ready for automation
2. **✅ Model works in JavaScript** → Compatible with React Native
3. **✅ Prediction shapes correct** → Architecture is valid

## 🚀 Ready for Automation

Once all tests pass, you can confidently implement:
- **Cloud Function auto-conversion**
- **CI/CD pipeline integration**
- **Automated model deployment**

## 🔧 Troubleshooting Failed Tests

If any test fails:

1. **Check the error messages** carefully
2. **Verify file paths** and permissions
3. **Test with a known working model** first
4. **Check model training logs** for issues

---

**Run these tests before implementing automation to ensure your pipeline will work correctly!**
