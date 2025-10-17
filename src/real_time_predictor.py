import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

class RealTimePredictor:
    """Real-time network traffic prediction for Intrusion Detection System"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.pca = None
        self.feature_columns = None
        self.models_loaded = False
        self.prediction_history = []
        
    def load_models(self, models_dir: str = "data/models/") -> bool:
        """Load trained models, scaler, and PCA transformer"""
        try:
            # Model name mapping: Display name -> File name
            model_mapping = {
                'Decision Tree (DT)': 'decision_tree_(dt)',
                'Random Forest (RF)': 'random_forest_(rf)',
                'Extra Trees (ET)': 'extra_trees_(et)',
                'XGBoost (XGB)': 'xgboost_(xgb)'
            }
            
            # Try to load all models with different possible naming conventions
            for display_name, file_name in model_mapping.items():
                # Try multiple file name patterns
                possible_names = [
                    f"{file_name}_model.pkl",
                    f"{display_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl",
                    f"{display_name.split('(')[0].strip().lower().replace(' ', '_')}_model.pkl"
                ]
                
                for possible_name in possible_names:
                    model_path = os.path.join(models_dir, possible_name)
                    if os.path.exists(model_path):
                        self.models[display_name] = joblib.load(model_path)
                        print(f"✓ Loaded {display_name}")
                        break
            
            # Load scaler
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Loaded scaler")
            
            # Load PCA
            pca_path = os.path.join(models_dir, "pca.pkl")
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                print("✓ Loaded PCA transformer")
            
            # Load feature columns
            features_path = os.path.join(models_dir, "feature_columns.pkl")
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
                print(f"✓ Loaded {len(self.feature_columns)} feature columns")
            
            self.models_loaded = len(self.models) > 0
            
            if not self.models_loaded:
                print("❌ No models found!")
                print(f"Looking for models in: {models_dir}")
                print("Expected file names:")
                for display_name, file_name in model_mapping.items():
                    print(f"  - {file_name}_model.pkl")
                
                # List actual files in directory
                if os.path.exists(models_dir):
                    actual_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                    print(f"\nActual files found: {actual_files}")
                
                return False
            
            print(f"\n✅ Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Preprocess raw network traffic data"""
        try:
            # Make a copy to avoid modifying original
            data = df.copy()
            
            # Remove Label column if present
            if 'Label' in data.columns:
                data = data.drop('Label', axis=1)
            
            # Ensure all required features are present
            if self.feature_columns is not None:
                # Add missing columns with 0
                for col in self.feature_columns:
                    if col not in data.columns:
                        data[col] = 0
                
                # Select only required features in correct order
                data = data[self.feature_columns]
            
            # Handle any remaining missing values
            data = data.fillna(0)
            
            # Replace infinite values
            data = data.replace([np.inf, -np.inf], 0)
            
            # Scale features
            if self.scaler is not None:
                try:
                    data_scaled = self.scaler.transform(data)
                except Exception as e:
                    print(f"Warning: Scaler transform failed: {e}")
                    data_scaled = data.values
            else:
                data_scaled = data.values
            
            # Apply PCA
            if self.pca is not None:
                try:
                    # Check if PCA is fitted
                    if hasattr(self.pca, 'transform'):
                        data_pca = self.pca.transform(data_scaled)
                    else:
                        print("Warning: PCA not fitted, using scaled data")
                        data_pca = data_scaled
                except Exception as e:
                    print(f"Warning: PCA transform failed: {e}")
                    data_pca = data_scaled
            else:
                # If no PCA, the data might already be PCA-transformed
                data_pca = data_scaled
            
            return data_pca
            
        except Exception as e:
            print(f"❌ Preprocessing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_single(self, data: pd.DataFrame, model_name: str = 'Random Forest') -> Dict:
        """Make prediction on single instance using specified model"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        try:
            # Preprocess
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return {"error": "Preprocessing failed"}
            
            # Get model
            model = self.models[model_name]
            
            # Predict
            prediction = model.predict(processed_data)[0]
            
            # Get prediction probability if available
            try:
                probabilities = model.predict_proba(processed_data)[0]
                confidence = float(max(probabilities))
                attack_probability = float(probabilities[1])
            except:
                confidence = 1.0
                attack_probability = float(prediction)
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'label': 'Attack' if prediction == 1 else 'Normal',
                'confidence': confidence,
                'attack_probability': attack_probability,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_ensemble(self, data: pd.DataFrame, voting: str = 'hard') -> Dict:
        """Make prediction using ensemble of all models"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            # Preprocess
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return {"error": "Preprocessing failed"}
            
            predictions = []
            probabilities = []
            model_results = {}
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                pred = model.predict(processed_data)[0]
                predictions.append(pred)
                
                try:
                    proba = model.predict_proba(processed_data)[0]
                    probabilities.append(proba[1])  # Attack probability
                    model_results[model_name] = {
                        'prediction': int(pred),
                        'attack_probability': float(proba[1])
                    }
                except:
                    probabilities.append(float(pred))
                    model_results[model_name] = {
                        'prediction': int(pred),
                        'attack_probability': float(pred)
                    }
            
            # Ensemble voting
            if voting == 'hard':
                # Majority voting
                final_prediction = int(np.round(np.mean(predictions)))
            else:
                # Soft voting (average probabilities)
                final_prediction = 1 if np.mean(probabilities) >= 0.5 else 0
            
            # Calculate confidence
            if voting == 'hard':
                confidence = float(predictions.count(final_prediction) / len(predictions))
            else:
                avg_prob = np.mean(probabilities)
                confidence = float(max(avg_prob, 1 - avg_prob))
            
            result = {
                'prediction': final_prediction,
                'label': 'Attack' if final_prediction == 1 else 'Normal',
                'confidence': confidence,
                'attack_probability': float(np.mean(probabilities)),
                'voting_method': voting,
                'individual_models': model_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ensemble prediction error: {str(e)}"}
    
    def predict_batch(self, data: pd.DataFrame, model_name: str = 'Random Forest') -> List[Dict]:
        """Make predictions on batch of instances"""
        if not self.models_loaded:
            return [{"error": "Models not loaded"}]
        
        if model_name not in self.models:
            return [{"error": f"Model '{model_name}' not found"}]
        
        try:
            # Preprocess
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return [{"error": "Preprocessing failed"}]
            
            # Get model
            model = self.models[model_name]
            
            # Predict
            predictions = model.predict(processed_data)
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(processed_data)
            except:
                probabilities = None
            
            # Prepare results
            results = []
            for i, pred in enumerate(predictions):
                if probabilities is not None:
                    confidence = float(max(probabilities[i]))
                    attack_prob = float(probabilities[i][1])
                else:
                    confidence = 1.0
                    attack_prob = float(pred)
                
                result = {
                    'index': i,
                    'prediction': int(pred),
                    'label': 'Attack' if pred == 1 else 'Normal',
                    'confidence': confidence,
                    'attack_probability': attack_prob,
                    'model_used': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{"error": f"Batch prediction error: {str(e)}"}]
    
    def add_to_history(self, prediction_result: Dict, original_data: pd.DataFrame = None):
        """Add prediction to history for monitoring"""
        history_entry = {
            'timestamp': prediction_result.get('timestamp', datetime.now().isoformat()),
            'prediction': prediction_result.get('prediction'),
            'label': prediction_result.get('label'),
            'confidence': prediction_result.get('confidence'),
            'attack_probability': prediction_result.get('attack_probability')
        }
        
        if original_data is not None and len(original_data) > 0:
            # Add some key features for analysis
            history_entry['features'] = original_data.iloc[0].to_dict()
        
        self.prediction_history.append(history_entry)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_statistics(self) -> Dict:
        """Get statistics from prediction history"""
        if not self.prediction_history:
            return {"message": "No predictions made yet"}
        
        df = pd.DataFrame(self.prediction_history)
        
        stats = {
            'total_predictions': len(df),
            'attacks_detected': int((df['prediction'] == 1).sum()),
            'normal_traffic': int((df['prediction'] == 0).sum()),
            'attack_rate': float((df['prediction'] == 1).mean()),
            'average_confidence': float(df['confidence'].mean()),
            'average_attack_probability': float(df['attack_probability'].mean())
        }
        
        return stats
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history = []
    
    def export_history(self, filepath: str = "data/predictions/prediction_history.csv"):
        """Export prediction history to CSV"""
        if not self.prediction_history:
            return False
        
        try:
            df = pd.DataFrame(self.prediction_history)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error exporting history: {str(e)}")
            return False