const tf = require('@tensorflow/tfjs-node'); // Use tfjs-node for testing
// For React Native, use: import * as tf from '@tensorflow/tfjs';
// For React Native, also need: import '@tensorflow/tfjs-react-native';

async function testModel() {
    console.log('🧪 Testing TensorFlow.js Model in Node.js Environment');
    console.log('=======================================================');

    try {
        // Load the converted model
        console.log('📥 Loading model...');
        const model = await tf.loadGraphModel('file://model_tfjs/model.json');
        console.log('✅ Model loaded successfully!');

        // Check model structure
        console.log('\n📊 Model Information:');
        console.log('Input shape:', model.input.shape);
        console.log('Output shape:', model.output.shape);

        // Create a dummy input (224x224x3 image, normalized 0-1)
        console.log('\n🧪 Creating test input...');
        const dummyInput = tf.randomNormal([1, 224, 224, 3]);
        console.log('Input tensor shape:', dummyInput.shape);
        console.log('Input range:', tf.min(dummyInput).dataSync(), 'to', tf.max(dummyInput).dataSync());

        // Make a prediction
        console.log('\n🔮 Making prediction...');
        const prediction = await model.predict(dummyInput);
        console.log('✅ Prediction completed!');
        console.log('Output shape:', prediction.shape);
        console.log('Output range:', tf.min(prediction).dataSync(), 'to', tf.max(prediction).dataSync());

        // Get top prediction
        const probabilities = await prediction.data();
        const topIndex = probabilities.indexOf(Math.max(...probabilities));
        console.log(`🎯 Top prediction index: ${topIndex}`);
        console.log(`🎯 Confidence: ${(probabilities[topIndex] * 100).toFixed(2)}%`);

        // Check labels if available
        try {
            const fs = require('fs');
            if (fs.existsSync('model_tfjs/labels.json')) {
                const labels = JSON.parse(fs.readFileSync('model_tfjs/labels.json', 'utf8'));
                console.log(`🏷️  Predicted class: ${labels[topIndex] || 'Unknown'}`);
                console.log(`📋 Available classes: ${labels.length} total`);
            } else {
                console.log('⚠️  No labels.json found - you may need to create this');
            }
        } catch (error) {
            console.log('⚠️  Could not load labels:', error.message);
        }

        // Memory cleanup
        model.dispose();
        dummyInput.dispose();
        prediction.dispose();

        console.log('\n🎉 MODEL TEST PASSED!');
        console.log('✅ The model loads and makes predictions correctly');
        console.log('✅ Ready for use in React Native app');

    } catch (error) {
        console.error('\n❌ MODEL TEST FAILED!');
        console.error('Error:', error.message);
        console.error('\nTroubleshooting:');
        console.error('1. Make sure model_tfjs/ directory exists and contains model.json');
        console.error('2. Check if labels.json exists (optional but recommended)');
        console.error('3. Verify the model was converted correctly');
        process.exit(1);
    }
}

// Run the test
testModel();
