#!/usr/bin/env python3
"""
Test script to verify that the linear model feature names fix works correctly
"""
import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from models.model_loader import ModelLoader
from utils.data_processor import DataProcessor

def test_linear_model_prediction():
    """Test linear model prediction with proper feature names"""
    print("Testing linear model prediction with feature names fix...")
    
    try:
        # Initialize components
        model_loader = ModelLoader()
        data_processor = DataProcessor()
        
        # Check if models are available
        available_models = model_loader.get_available_models()
        if not available_models:
            print("❌ No models found. Please train models first.")
            return False
        
        print(f"✅ Found {len(available_models)} models:")
        for model_name, info in available_models.items():
            print(f"   - {model_name}: {info['type']}")
        
        # Create sample data
        sample_data = data_processor.create_sample_data()
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        
        # Process the data
        X_processed = data_processor.process_data(sample_data)
        print(f"✅ Processed data shape: {X_processed.shape}")
        print(f"✅ Feature names: {list(X_processed.columns[:5])}...")  # Show first 5
        
        # Test with linear model if available
        linear_model_name = None
        for model_name, info in available_models.items():
            if info['type'] in ['linear', 'logistic']:
                linear_model_name = model_name
                break
        
        if not linear_model_name:
            print("❌ No linear model found for testing")
            return False
        
        print(f"✅ Testing with model: {linear_model_name}")
        
        # Test prediction with DataFrame (should work without warnings)
        print("\n--- Testing with DataFrame input ---")
        predictions_df = model_loader.predict(linear_model_name, X_processed)
        probabilities_df = model_loader.predict_proba(linear_model_name, X_processed)
        
        print(f"✅ Predictions shape: {predictions_df.shape}")
        print(f"✅ Probabilities shape: {probabilities_df.shape}")
        print(f"✅ Predictions: {predictions_df}")
        print(f"✅ Probabilities: {probabilities_df}")
        
        # Test prediction with numpy array (should also work now)
        print("\n--- Testing with numpy array input ---")
        predictions_np = model_loader.predict(linear_model_name, X_processed.values)
        probabilities_np = model_loader.predict_proba(linear_model_name, X_processed.values)
        
        print(f"✅ Predictions shape: {predictions_np.shape}")
        print(f"✅ Probabilities shape: {probabilities_np.shape}")
        print(f"✅ Predictions: {predictions_np}")
        print(f"✅ Probabilities: {probabilities_np}")
        
        # Verify results are the same
        if np.array_equal(predictions_df, predictions_np) and np.array_equal(probabilities_df, probabilities_np):
            print("✅ Results are identical between DataFrame and numpy array inputs")
        else:
            print("❌ Results differ between DataFrame and numpy array inputs")
            return False
        
        print("\n🎉 All tests passed! The feature names fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_linear_model_prediction()
    sys.exit(0 if success else 1) 