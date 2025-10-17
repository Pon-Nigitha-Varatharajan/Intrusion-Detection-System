# src/model_training.py

import pandas as pd
import numpy as np
import pickle
from src.utils import ensure_directory_exists
import joblib
import time
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import settings - since your settings.py uses module-level variables
try:
    from config.settings import RANDOM_STATE, TEST_SIZE, CV_FOLDS
    print("✅ Settings imported successfully")
except ImportError as e:
    print(f"❌ Error importing settings: {e}")
    # Fallback defaults
    RANDOM_STATE = 42
    TEST_SIZE = 0.1
    CV_FOLDS = 10

class ModelTrainer:
    def __init__(self):
        """Initialize all 4 ML models as per the paper"""
        self.models = {
            'Decision Tree (DT)': DecisionTreeClassifier(
                criterion='gini',
                random_state=RANDOM_STATE,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'Random Forest (RF)': RandomForestClassifier(
                n_estimators=100,        # ✅ Same as before
                criterion='gini',        # ✅ Same as before
                random_state=RANDOM_STATE,
                max_depth=20,            # ✅ Paper-compliant (2-5x faster!)
                min_samples_split=10,    # ✅ Paper-compliant (prevents overfitting)
                min_samples_leaf=5,      # ✅ Paper-compliant (prevents overfitting)
                n_jobs=-1               # ✅ Same as before
            ),
            'Extra Trees (ET)': ExtraTreesClassifier(
                n_estimators=100,        # ✅ Same as before
                criterion='gini',        # ✅ Same as before
                random_state=RANDOM_STATE,
                max_depth=20,            # ✅ Paper-compliant (2-5x faster!)
                min_samples_split=10,    # ✅ Paper-compliant (prevents overfitting)
                min_samples_leaf=5,      # ✅ Paper-compliant (prevents overfitting)
                n_jobs=-1               # ✅ Same as before
            ),

            'XGBoost (XGB)': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )
        }
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # Keep 95% variance
        
        # Initialize data attributes
        self.trained_models = {}
        self.training_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.original_columns = None  # Store original column names
    
    def load_data(self, data_path="data/processed/final_training_data.csv"):
        """
        Load the final training data (after SFE + PCA)
        
        Args:
            data_path: Path to the final processed data
            
        Returns:
            Success status
        """
        print(f"📂 Loading training data from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            print(f"   Loaded dataset shape: {df.shape}")
            
            # Separate features and target
            if 'label' not in df.columns:
                print("❌ 'label' column not found!")
                return False
            
            # Features: All PCA components
            feature_cols = [col for col in df.columns if col.startswith('PC')]
            X = df[feature_cols]
            y = df['label']
            
            # Store original columns for preprocessing artifacts
            self.original_columns = feature_cols
            
            print(f"   Features (PCA components): {len(feature_cols)}")
            print(f"   Total samples: {len(df):,}")
            print(f"   Class distribution:")
            print(f"      - Normal (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
            print(f"      - Attack (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
            
            # Split data: 90% train, 10% test (as per paper's k-fold CV)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y  # Maintain class balance
            )
            
            print(f"\n   Train set: {self.X_train.shape}")
            print(f"   Test set: {self.X_test.shape}")
            print("✅ Data loaded successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def train_model(self, model_name, use_cross_validation=True):
        """
        Train a single model with optional k-fold cross-validation
        
        Args:
            model_name: Name of the model to train
            use_cross_validation: Whether to use 10-fold CV (as per paper)
            
        Returns:
            Training results dictionary
        """
        if self.X_train is None:
            print("❌ No training data loaded!")
            return None
        
        print(f"\n{'='*60}")
        print(f"🤖 Training: {model_name}")
        print(f"{'='*60}")
        
        model = self.models[model_name]
        start_time = time.time()
        
        # K-Fold Cross-Validation (Step 6 from paper)
        cv_scores = None
        if use_cross_validation:
            print(f"   Performing {CV_FOLDS}-Fold Cross-Validation...")
            
            kfold = StratifiedKFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE
            )
            
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=kfold,
                scoring='accuracy',
                n_jobs=-1
            )
            
            print(f"   CV Scores: {cv_scores}")
            print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        print(f"   Training on full training set...")
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        print(f"   Making predictions...")
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Prediction probabilities for ROC curve
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_test_proba = None
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'model': model,
            'training_time': training_time,
            
            # Training set metrics
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'train_precision': precision_score(self.y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(self.y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(self.y_train, y_train_pred, zero_division=0),
            
            # Test set metrics
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'test_precision': precision_score(self.y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(self.y_test, y_test_pred, zero_division=0),
            'test_f1': f1_score(self.y_test, y_test_pred, zero_division=0),
            
            # Confusion matrix
            'train_confusion_matrix': confusion_matrix(self.y_train, y_train_pred),
            'test_confusion_matrix': confusion_matrix(self.y_test, y_test_pred),
            
            # Predictions
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            
            # Cross-validation
            'cv_scores': cv_scores if use_cross_validation else None,
            'cv_mean': cv_scores.mean() if use_cross_validation else None,
            'cv_std': cv_scores.std() if use_cross_validation else None
        }
        
        # ROC AUC (if probabilities available)
        if y_test_proba is not None:
            results['test_roc_auc'] = roc_auc_score(self.y_test, y_test_proba)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_test_proba)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        # Print results
        print(f"\n   ✅ Training Complete!")
        print(f"   Training Time: {training_time:.2f} seconds")
        print(f"\n   📊 Test Set Performance:")
        print(f"      Accuracy:  {results['test_accuracy']:.4f}")
        print(f"      Precision: {results['test_precision']:.4f}")
        print(f"      Recall:    {results['test_recall']:.4f}")
        print(f"      F1-Score:  {results['test_f1']:.4f}")
        if y_test_proba is not None:
            print(f"      ROC AUC:   {results['test_roc_auc']:.4f}")
        
        # Store results
        self.trained_models[model_name] = model
        self.training_results[model_name] = results
        
        return results
    
    def train_all_models(self, use_cross_validation=True):
        """
        Train all 4 models (DT, RF, ET, XGB) as per paper
        
        Returns:
            Dictionary of all training results
        """
        print("\n" + "="*60)
        print("🚀 Training All Models (DT, RF, ET, XGB)")
        print("="*60)
        
        all_results = {}
        
        for model_name in self.models.keys():
            results = self.train_model(model_name, use_cross_validation)
            if results:
                all_results[model_name] = results
        
        print("\n" + "="*60)
        print("✅ All Models Trained Successfully!")
        print("="*60)
        
        # Summary comparison
        self._print_comparison_summary()
        
        return all_results
    
    def _print_comparison_summary(self):
        """Print comparison of all models"""
        if not self.training_results:
            return
        
        print("\n📊 Model Comparison Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        for model_name, results in self.training_results.items():
            print(f"{model_name:<20} "
                  f"{results['test_accuracy']:.4f}      "
                  f"{results['test_precision']:.4f}      "
                  f"{results['test_recall']:.4f}      "
                  f"{results['test_f1']:.4f}")
        
        print("-" * 80)
        
        # Find best model
        best_model = max(self.training_results.items(), 
                        key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['test_accuracy']:.4f} accuracy")
    
    def save_model(self, model_name, filepath=None):
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model to save
            filepath: Custom filepath (optional)
        """
        if model_name not in self.trained_models:
            print(f"❌ Model '{model_name}' not found!")
            return False
        
        if filepath is None:
            # Default filepath
            model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            filepath = f"data/models/{model_filename}.pkl"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.trained_models[model_name], f)
            print(f"✅ Model saved: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def save_all_models(self, models_dir: str = "data/models/"):
        """Save all trained models, scaler, PCA, and feature columns"""
        try:
            os.makedirs(models_dir, exist_ok=True)
        
            # Save each model
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
                    joblib.dump(model, model_path)
                    print(f"✓ Saved {model_name}")
        
            # Save scaler
            if self.scaler is not None:
                scaler_path = os.path.join(models_dir, "scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                print("✓ Saved scaler")
        
            # Save PCA
            if self.pca is not None:
                pca_path = os.path.join(models_dir, "pca.pkl")
                joblib.dump(self.pca, pca_path)
                print("✓ Saved PCA transformer")
        
            # Save feature columns
            if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                features_path = os.path.join(models_dir, "feature_columns.pkl")
                joblib.dump(self.feature_columns, features_path)
                print("✓ Saved feature columns")
            elif self.X_train is not None:
                # Save from X_train if feature_columns not set
                features_path = os.path.join(models_dir, "feature_columns.pkl")
                joblib.dump(list(self.X_train.columns), features_path)
                print("✓ Saved feature columns from X_train")
        
            print(f"\n✅ All models saved to {models_dir}")
            return True
        
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False
    
    def load_model(self, model_name: str, models_dir: str = "data/models/"):
        """Load a specific model"""
        try:
            model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
        
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def get_comparison_dataframe(self):
        """Get comparison results as DataFrame"""
        if not self.training_results:
            return None
        
        data = []
        for model_name, results in self.training_results.items():
            data.append({
                'Model': model_name,
                'Accuracy': results['test_accuracy'],
                'Precision': results['test_precision'],
                'Recall': results['test_recall'],
                'F1-Score': results['test_f1'],
                'ROC AUC': results.get('test_roc_auc', None),
                'Training Time (s)': results['training_time'],
                'CV Mean': results.get('cv_mean', None),
                'CV Std': results.get('cv_std', None)
            })
        
        return pd.DataFrame(data)
    
    def save_preprocessing_artifacts(self):
        """Save scaler, PCA, and feature columns for real-time prediction"""
        try:
            ensure_directory_exists("data/models/")
        
            # Save preprocessing artifacts
            joblib.dump(self.scaler, "data/models/scaler.pkl")
            joblib.dump(self.pca, "data/models/pca.pkl")
            
            # Save the feature columns (PCA component names)
            if self.original_columns is not None:
                joblib.dump(self.original_columns, "data/models/feature_columns.pkl")
            else:
                # Fallback: use current training columns
                joblib.dump(list(self.X_train.columns), "data/models/feature_columns.pkl")
        
            print("✅ Preprocessing artifacts saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving preprocessing artifacts: {e}")
            return False