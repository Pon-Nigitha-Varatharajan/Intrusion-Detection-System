"""
Generate test data that matches your trained model's expected format
This creates PCA-transformed data that your models can actually use
"""

import pandas as pd
import numpy as np
import joblib
import os

def load_training_reference():
    """Load reference data to understand the expected format"""
    train_path = "data/processed/final_training_data.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return None
    
    df = pd.read_csv(train_path)
    print(f"✓ Loaded training data: {df.shape}")
    
    # Find label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if label_col is None:
        print("❌ No label column found!")
        return None
    
    # Separate features and labels
    # Keep only PCA features, remove label and attack_cat
    exclude_cols = [label_col]
    if 'attack_cat' in df.columns:
        exclude_cols.append('attack_cat')
    
    X = df.drop(columns=exclude_cols)
    y = df[label_col]
    
    print(f"✓ Features: {list(X.columns)}")
    print(f"✓ Feature count: {len(X.columns)}")
    print(f"✓ Label column: '{label_col}'")
    print(f"✓ Label distribution:")
    print(f"   Normal (0): {(y == 0).sum()}")
    print(f"   Attack (1): {(y == 1).sum()}")
    
    # Check if PCA transformed
    is_pca = X.columns[0].startswith('PC')
    print(f"✓ Is PCA transformed: {is_pca}")
    
    return {
        'features': X,
        'labels': y,
        'columns': list(X.columns),
        'is_pca': is_pca,
        'shape': X.shape,
        'label_col': label_col
    }

def generate_pca_test_data(n_samples=100, attack_ratio=0.3, reference_data=None):
    """Generate test data in PCA space that matches training data"""
    
    if reference_data is None:
        reference_data = load_training_reference()
    
    if reference_data is None:
        print("❌ Cannot generate test data without reference")
        return None
    
    X_ref = reference_data['features']
    y_ref = reference_data['labels']
    
    if y_ref is None:
        print("❌ No labels found")
        return None
    
    n_features = X_ref.shape[1]
    
    print(f"\n📊 Generating {n_samples} test samples with {n_features} features...")
    
    # Separate normal and attack samples from reference
    normal_samples = X_ref[y_ref == 0]
    attack_samples = X_ref[y_ref == 1]
    
    print(f"✓ Reference - Normal: {len(normal_samples)}, Attack: {len(attack_samples)}")
    
    if len(normal_samples) == 0 or len(attack_samples) == 0:
        print("❌ Need both normal and attack samples")
        return None
    
    # Calculate number of each type to generate
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    test_data = []
    test_labels = []
    
    # Generate normal samples by adding noise to reference normal samples
    for _ in range(n_normal):
        # Pick a random normal sample
        base_sample = normal_samples.sample(1).values[0]
        
        # Add small Gaussian noise (preserving the structure)
        noise = np.random.normal(0, 0.1, n_features)
        new_sample = base_sample + noise
        
        test_data.append(new_sample)
        test_labels.append(0)
    
    # Generate attack samples by adding noise to reference attack samples
    for _ in range(n_attacks):
        # Pick a random attack sample
        base_sample = attack_samples.sample(1).values[0]
        
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.1, n_features)
        new_sample = base_sample + noise
        
        test_data.append(new_sample)
        test_labels.append(1)
    
    # Create DataFrame with 'Label' column (uppercase)
    df = pd.DataFrame(test_data, columns=reference_data['columns'])
    df['Label'] = test_labels
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✓ Generated {n_normal} normal and {n_attacks} attack samples")
    
    return df

def generate_from_training_distribution(n_samples=100, attack_ratio=0.3):
    """
    Generate samples by sampling from the actual training data distribution
    This is the most reliable method
    """
    reference_data = load_training_reference()
    
    if reference_data is None:
        return None
    
    X_ref = reference_data['features']
    y_ref = reference_data['labels']
    
    if y_ref is None:
        print("❌ No labels found in training data")
        return None
    
    # Calculate samples needed
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks
    
    # Sample directly from training data
    normal_indices = y_ref[y_ref == 0].index
    attack_indices = y_ref[y_ref == 1].index
    
    print(f"✓ Available - Normal: {len(normal_indices)}, Attack: {len(attack_indices)}")
    
    if len(normal_indices) == 0 or len(attack_indices) == 0:
        print("❌ Need both normal and attack samples in training data")
        return None
    
    # Randomly sample with replacement
    sampled_normal_idx = np.random.choice(normal_indices, n_normal, replace=True)
    sampled_attack_idx = np.random.choice(attack_indices, n_attacks, replace=True)
    
    # Combine
    all_indices = list(sampled_normal_idx) + list(sampled_attack_idx)
    
    # Get the data
    X_sampled = X_ref.loc[all_indices].reset_index(drop=True)
    y_sampled = y_ref.loc[all_indices].reset_index(drop=True)
    
    # Combine into DataFrame with 'Label' column (uppercase for consistency)
    df = X_sampled.copy()
    df['Label'] = y_sampled.values
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✓ Sampled {n_normal} normal and {n_attacks} attack samples")
    
    return df

def save_test_files():
    """Generate and save various test files"""
    
    print("="*60)
    print("GENERATING TEST DATA FILES")
    print("="*60)
    
    # Ensure directory exists
    os.makedirs('data/test', exist_ok=True)
    
    # Load reference once
    reference = load_training_reference()
    
    if reference is None:
        print("\n❌ Cannot generate test files without training data")
        print("Please ensure data/processed/final_training_data.csv exists")
        return
    
    print(f"\n{'='*60}")
    
    # 1. Small test file (for single predictions)
    print("\n1️⃣ Generating small test file (10 samples)...")
    df_small = generate_from_training_distribution(n_samples=10, attack_ratio=0.3)
    df_small.to_csv('data/test/test_small.csv', index=False)
    print(f"✅ Saved: data/test/test_small.csv")
    print(f"   Normal: {(df_small['Label'] == 0).sum()}, Attack: {(df_small['Label'] == 1).sum()}")
    
    # 2. Medium test file (for batch predictions)
    print("\n2️⃣ Generating medium test file (100 samples)...")
    df_medium = generate_from_training_distribution(n_samples=100, attack_ratio=0.3)
    df_medium.to_csv('data/test/test_medium.csv', index=False)
    print(f"✅ Saved: data/test/test_medium.csv")
    print(f"   Normal: {(df_medium['Label'] == 0).sum()}, Attack: {(df_medium['Label'] == 1).sum()}")
    
    # 3. Large test file (for streaming)
    print("\n3️⃣ Generating large test file (500 samples)...")
    df_large = generate_from_training_distribution(n_samples=500, attack_ratio=0.2)
    df_large.to_csv('data/test/test_large.csv', index=False)
    print(f"✅ Saved: data/test/test_large.csv")
    print(f"   Normal: {(df_large['Label'] == 0).sum()}, Attack: {(df_large['Label'] == 1).sum()}")
    
    # 4. Mostly normal traffic
    print("\n4️⃣ Generating normal traffic file (200 samples, 5% attacks)...")
    df_normal = generate_from_training_distribution(n_samples=200, attack_ratio=0.05)
    df_normal.to_csv('data/test/test_normal_traffic.csv', index=False)
    print(f"✅ Saved: data/test/test_normal_traffic.csv")
    print(f"   Normal: {(df_normal['Label'] == 0).sum()}, Attack: {(df_normal['Label'] == 1).sum()}")
    
    # 5. Attack scenario
    print("\n5️⃣ Generating attack scenario (200 samples, 70% attacks)...")
    df_attack = generate_from_training_distribution(n_samples=200, attack_ratio=0.7)
    df_attack.to_csv('data/test/test_attack_scenario.csv', index=False)
    print(f"✅ Saved: data/test/test_attack_scenario.csv")
    print(f"   Normal: {(df_attack['Label'] == 0).sum()}, Attack: {(df_attack['Label'] == 1).sum()}")
    
    # 6. Balanced dataset
    print("\n6️⃣ Generating balanced dataset (100 samples, 50-50)...")
    df_balanced = generate_from_training_distribution(n_samples=100, attack_ratio=0.5)
    df_balanced.to_csv('data/test/test_balanced.csv', index=False)
    print(f"✅ Saved: data/test/test_balanced.csv")
    print(f"   Normal: {(df_balanced['Label'] == 0).sum()}, Attack: {(df_balanced['Label'] == 1).sum()}")
    
    print("\n" + "="*60)
    print("✅ ALL TEST FILES GENERATED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📁 Test files created in data/test/:")
    print("   - test_small.csv (10 samples)")
    print("   - test_medium.csv (100 samples)")
    print("   - test_large.csv (500 samples)")
    print("   - test_normal_traffic.csv (mostly normal)")
    print("   - test_attack_scenario.csv (mostly attacks)")
    print("   - test_balanced.csv (50-50 split)")
    
    print("\n💡 Usage:")
    print("   1. Upload any of these files to the Real-Time Prediction page")
    print("   2. They will work correctly with your trained models")
    print("   3. You can verify predictions match the actual labels")

def quick_test():
    """Quick test to verify generated data works with predictor"""
    print("\n" + "="*60)
    print("QUICK PREDICTION TEST")
    print("="*60)
    
    try:
        from src.real_time_predictor import RealTimePredictor
        
        # Generate small test
        df_test = generate_from_training_distribution(n_samples=5, attack_ratio=0.4)
        
        if df_test is None:
            return
        
        # Separate features and labels
        X_test = df_test.drop('Label', axis=1)
        y_true = df_test['Label']
        
        print(f"\n✓ Generated {len(df_test)} test samples")
        print(f"  Actual labels: {y_true.values}")
        
        # Load predictor
        predictor = RealTimePredictor()
        if not predictor.load_models():
            print("\n❌ Could not load models")
            return
        
        print("\n🔮 Making predictions...")
        
        # Make predictions
        results = predictor.predict_batch(X_test, model_name='Random Forest (RF)')
        
        if results and 'error' not in results[0]:
            predictions = [r['prediction'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            print(f"✓ Predictions: {predictions}")
            print(f"✓ Confidences: {[f'{c:.2f}' for c in confidences]}")
            
            # Calculate accuracy
            correct = sum([p == t for p, t in zip(predictions, y_true)])
            accuracy = correct / len(predictions)
            
            print(f"\n✅ Accuracy: {accuracy*100:.1f}% ({correct}/{len(predictions)} correct)")
        else:
            print(f"\n❌ Prediction failed: {results}")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DATA GENERATOR FOR INTRUSION DETECTION")
    print("="*60)
    
    # Generate all test files
    save_test_files()
    
    # Run quick test
    quick_test()
    
    print("\n✅ Done! You can now use the test files for predictions.")