import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score

def diagnose_model_issue():
    print("🔍 MODEL DIAGNOSIS STARTED")
    print("=" * 60)
    
    models_dir = "data/models/"
    
    # Check what files exist
    print("📁 Checking model files...")
    if not os.path.exists(models_dir):
        print("❌ Models directory doesn't exist!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        file_path = os.path.join(models_dir, file)
        file_size = os.path.getsize(file_path) / 1024
        print(f"   📄 {file} ({file_size:.1f} KB)")
    
    # Test loading one model
    print("\n🧪 Testing model loading...")
    try:
        # Try to load decision tree
        dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
        if os.path.exists(dt_path):
            dt_model = joblib.load(dt_path)
            print(f"✅ Decision Tree loaded successfully")
            
            # Check model properties
            if hasattr(dt_model, 'classes_'):
                print(f"   Model classes: {dt_model.classes_}")
                print(f"   Number of classes: {len(dt_model.classes_)}")
                
                # Check if it's binary classification
                if len(dt_model.classes_) == 2:
                    print(f"   ✅ Binary classification: {dt_model.classes_}")
                else:
                    print(f"   ❌ Unexpected number of classes: {dt_model.classes_}")
                    
                # Check feature importance
                if hasattr(dt_model, 'feature_importances_'):
                    print(f"   Feature importance shape: {dt_model.feature_importances_.shape}")
                    print(f"   Max feature importance: {dt_model.feature_importances_.max():.4f}")
            else:
                print("   ❌ Model doesn't have classes_ attribute - may not be trained!")
                
            # Test prediction on dummy data
            print("\n🧪 Testing model prediction...")
            dummy_data = np.random.randn(1, 10)  # Assuming 10 features
            try:
                prediction = dt_model.predict(dummy_data)
                probabilities = dt_model.predict_proba(dummy_data)
                print(f"   Dummy prediction: {prediction[0]}")
                print(f"   Probabilities: {probabilities[0]}")
                print(f"   Probability sum: {probabilities[0].sum():.2f}")
                
                # Check if probabilities make sense
                if np.all(probabilities[0] > 0.9) or np.all(probabilities[0] < 0.1):
                    print("   ⚠️  Suspicious probabilities detected!")
                    
            except Exception as e:
                print(f"   ❌ Prediction test failed: {e}")
                
        else:
            print("❌ Decision Tree model not found!")
            
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
    
    # Check preprocessing artifacts
    print("\n🔧 Checking preprocessing artifacts...")
    artifacts = ['scaler.pkl', 'pca.pkl', 'feature_columns.pkl']
    for artifact in artifacts:
        artifact_path = os.path.join(models_dir, artifact)
        if os.path.exists(artifact_path):
            try:
                artifact_obj = joblib.load(artifact_path)
                if artifact == 'scaler.pkl':
                    print(f"✅ Scaler: {type(artifact_obj)}")
                elif artifact == 'pca.pkl':
                    print(f"✅ PCA: {type(artifact_obj)}")
                    if hasattr(artifact_obj, 'n_components_'):
                        print(f"   PCA components: {artifact_obj.n_components_}")
                elif artifact == 'feature_columns.pkl':
                    print(f"✅ Feature columns: {len(artifact_obj)} features")
                    print(f"   First 5 features: {artifact_obj[:5]}")
            except Exception as e:
                print(f"❌ Error loading {artifact}: {e}")
        else:
            print(f"❌ {artifact} not found!")

if __name__ == "__main__":
    diagnose_model_issue()