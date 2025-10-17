import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from config import settings

def ensure_directory_exists(file_path):
    """Ensure directory exists for file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def load_combined_dataset():
    """Load UNSW-NB15 dataset as per paper specification - ONLY training and testing sets"""
    try:
        data_dir = settings.DATA_RAW_PATH
        
        print(f"🔍 Looking for UNSW-NB15 files in: {data_dir}")
        
        # Load ONLY training and testing sets as per paper (Page 12)
        training_file = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
        testing_file = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
        
        dfs = []
        
        # Load training set (175,341 records as per paper)
        if os.path.exists(training_file):
            df_train = pd.read_csv(training_file)
            print(f"✅ Loaded training set: {df_train.shape}")
            print(f"   Training set columns: {len(df_train.columns)}")
            dfs.append(df_train)
        else:
            print(f"❌ Training set not found: {training_file}")
            return None
        
        # Load testing set (82,332 records as per paper)  
        if os.path.exists(testing_file):
            df_test = pd.read_csv(testing_file)
            print(f"✅ Loaded testing set: {df_test.shape}")
            print(f"   Testing set columns: {len(df_test.columns)}")
            dfs.append(df_test)
        else:
            print(f"❌ Testing set not found: {testing_file}")
            return None
        
        # Combine only training and testing (Total: 257,673 records as per paper)
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"✅ Combined dataset shape: {combined_df.shape}")
        print(f"📊 Paper's expected total: 257,673 records")
        print(f"📊 Our combined dataset: {len(combined_df)} records")
        
        # Verify against paper's numbers
        if len(combined_df) == 257673:
            print("🎯 Perfect! Exactly matches paper's dataset size: 257,673 records")
        else:
            print(f"⚠️ Note: Dataset size {len(combined_df)} vs paper's 257,673")
        
        # Show class distribution
        if 'label' in combined_df.columns:
            label_counts = combined_df['label'].value_counts()
            print(f"📊 Class distribution - Normal: {label_counts.get(0, 0)}, Attack: {label_counts.get(1, 0)}")
        
        return combined_df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_processed_data(file_path):
    """
    Load processed data from specified file path
    Returns DataFrame or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded processed data from {file_path}: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def get_data_info(df):
    """
    Get basic information about the dataset
    """
    if df is None or len(df) == 0:
        return {"error": "Empty or None dataframe"}
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    # Add target distribution if label column exists
    if 'label' in df.columns:
        info['label_distribution'] = df['label'].value_counts().to_dict()
    
    # Add attack category distribution if exists
    if 'attack_cat' in df.columns:
        info['attack_category_distribution'] = df['attack_cat'].value_counts().to_dict()
    
    return info

def save_dataframe(df, file_path, index=False):
    """
    Save DataFrame to CSV file
    """
    try:
        ensure_directory_exists(file_path)
        df.to_csv(file_path, index=index)
        print(f"✅ Saved DataFrame to {file_path}: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Error saving to {file_path}: {e}")
        return False

def save_model(model, filename):
    """Save trained model"""
    ensure_directory_exists(settings.MODELS_PATH)
    filepath = os.path.join(settings.MODELS_PATH, filename)
    joblib.dump(model, filepath)

def load_model(filename):
    """Load trained model"""
    filepath = os.path.join(settings.MODELS_PATH, filename)
    return joblib.load(filepath)

def get_file_size(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return round(os.path.getsize(file_path) / (1024 * 1024), 2)
    return 0

def check_file_exists(file_path):
    """
    Check if file exists and return file info
    """
    if not os.path.exists(file_path):
        return {
            'exists': False,
            'size_mb': 0,
            'message': f"File not found: {file_path}"
        }
    
    file_size = get_file_size(file_path)
    
    # For CSV files, also check if we can load them
    if file_path.endswith('.csv'):
        try:
            df_sample = pd.read_csv(file_path, nrows=5)
            row_count = "Unknown"
            # Try to get approximate row count without loading entire file
            with open(file_path, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            
            return {
                'exists': True,
                'size_mb': file_size,
                'rows': row_count,
                'columns': len(df_sample.columns),
                'message': f"✅ CSV file: {row_count} rows, {len(df_sample.columns)} columns"
            }
        except Exception as e:
            return {
                'exists': True,
                'size_mb': file_size,
                'message': f"⚠️ File exists but cannot read: {e}"
            }
    
    return {
        'exists': True,
        'size_mb': file_size,
        'message': f"✅ File exists: {file_path}"
    }

def get_dataset_files_info():
    """
    Get information about all dataset files in the project
    """
    files_info = {}
    
    # Raw data files
    import glob
    raw_files = glob.glob("data/raw/*.csv")
    files_info['raw_data'] = [check_file_exists(f) for f in raw_files]
    
    # Processed data files
    processed_files = [
        "data/processed/preprocessed_data.csv",
        "data/processed/normalized_data.csv", 
        "data/processed/balanced_data.csv",
        "data/processed/engineered_features.csv"
    ]
    files_info['processed_data'] = [check_file_exists(f) for f in processed_files]
    
    # Model files
    model_files = glob.glob("models/*.joblib") + glob.glob("models/*.pkl")
    files_info['model_files'] = [check_file_exists(f) for f in model_files]
    
    return files_info

def clean_column_names(df):
    """
    Clean column names: remove spaces, special characters, etc.
    """
    df_clean = df.copy()
    df_clean.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df_clean.columns]
    return df_clean

def get_missing_value_summary(df):
    """
    Get detailed missing value summary
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'column': missing_count.index,
        'missing_count': missing_count.values,
        'missing_percentage': missing_percentage.values
    })
    
    missing_summary = missing_summary[missing_summary['missing_count'] > 0]\
        .sort_values('missing_percentage', ascending=False)
    
    return missing_summary

def optimize_data_types(df):
    """
    Optimize data types to reduce memory usage
    """
    df_opt = df.copy()
    
    # Optimize numerical columns
    numerical_cols = df_opt.select_dtypes(include=['int64']).columns
    for col in numerical_cols:
        if df_opt[col].min() >= 0:  # Unsigned integers
            if df_opt[col].max() < 256:
                df_opt[col] = df_opt[col].astype('uint8')
            elif df_opt[col].max() < 65536:
                df_opt[col] = df_opt[col].astype('uint16')
            elif df_opt[col].max() < 4294967296:
                df_opt[col] = df_opt[col].astype('uint32')
        else:  # Signed integers
            if df_opt[col].min() >= -128 and df_opt[col].max() < 128:
                df_opt[col] = df_opt[col].astype('int8')
            elif df_opt[col].min() >= -32768 and df_opt[col].max() < 32768:
                df_opt[col] = df_opt[col].astype('int16')
            elif df_opt[col].min() >= -2147483648 and df_opt[col].max() < 2147483648:
                df_opt[col] = df_opt[col].astype('int32')
    
    # Optimize float columns
    float_cols = df_opt.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_opt[col] = df_opt[col].astype('float32')
    
    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    optimized_memory = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = ((original_memory - optimized_memory) / original_memory) * 100
    
    print(f"💾 Memory optimization: {original_memory:.2f}MB → {optimized_memory:.2f}MB ({reduction:.1f}% reduction)")
    
    return df_opt

# =============================================================================
# NEW FUNCTIONS ADDED FOR REAL-TIME PREDICTION
# =============================================================================

def save_prediction_log(prediction_result, log_file="data/predictions/prediction_log.json"):
    """Save prediction results to log file"""
    ensure_directory_exists(log_file)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    prediction_result = convert_numpy_types(prediction_result)
    
    try:
        # Read existing log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Append new prediction
        log_data.append(prediction_result)
        
        # Keep only last 1000 predictions
        if len(log_data) > 1000:
            log_data = log_data[-1000:]
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        return True
    except Exception as e:
        print(f"❌ Error saving prediction log: {e}")
        return False

def load_prediction_log(log_file="data/predictions/prediction_log.json"):
    """Load prediction log"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading prediction log: {e}")
        return []

def format_prediction_result(result):
    """Format prediction result for display"""
    if not result:
        return "❌ Error in prediction"
    
    try:
        formatted = f"""
⏰ **Timestamp**: {result.get('timestamp', 'Unknown')}
🎯 **Final Verdict**: {result.get('final_verdict', 'Unknown')}
📊 **Ensemble Confidence**: {result.get('ensemble_confidence', 0):.2%}

📈 **Individual Model Predictions**:
"""
        
        predictions = result.get('predictions', {})
        confidences = result.get('confidences', {})
        
        for model_name, prediction in predictions.items():
            confidence = confidences.get(model_name, 0)
            verdict = "🚨 Attack" if prediction == 1 else "✅ Normal" if prediction == 0 else "⚠️ Error"
            formatted += f"    • **{model_name}**: {verdict} (Confidence: {confidence:.2%})\n"
        
        return formatted
        
    except Exception as e:
        return f"❌ Error formatting prediction result: {e}"

def create_sample_network_traffic(traffic_type="normal"):
    """
    Create sample network traffic data for testing predictions
    Based on UNSW-NB15 dataset features
    
    Parameters:
    - traffic_type: "normal" or "attack"
    """
    if traffic_type == "normal":
        return {
            'dur': np.random.uniform(0, 10),
            'proto': np.random.choice([0, 1, 2]),  # TCP, UDP, ICMP
            'service': np.random.choice([0, 1, 2, 3]),  # HTTP, FTP, SSH, DNS
            'state': np.random.choice([0, 1, 2, 3]),  # SF, S0, REJ, RSTO
            'spkts': np.random.randint(1, 100),
            'dpkts': np.random.randint(1, 100),
            'sbytes': np.random.randint(0, 1000),
            'dbytes': np.random.randint(0, 1000),
            'rate': np.random.uniform(0, 1),
            'sttl': np.random.randint(30, 255),
            'dttl': np.random.randint(30, 255),
            'sload': np.random.uniform(0, 1000),
            'dload': np.random.uniform(0, 1000),
            'sloss': np.random.randint(0, 10),
            'dloss': np.random.randint(0, 10),
            'sinpkt': np.random.uniform(0, 10),
            'dinpkt': np.random.uniform(0, 10),
            'sjit': np.random.uniform(0, 1),
            'djit': np.random.uniform(0, 1),
            'swin': np.random.randint(0, 65535),
            'stcpb': np.random.randint(0, 1000000),
            'dtcpb': np.random.randint(0, 1000000),
            'dwin': np.random.randint(0, 65535),
            'tcprtt': np.random.uniform(0, 10),
            'synack': np.random.uniform(0, 10),
            'ackdat': np.random.uniform(0, 10),
            'smean': np.random.randint(1, 100),
            'dmean': np.random.randint(1, 100),
            'trans_depth': np.random.randint(0, 5),
            'response_body_len': np.random.randint(0, 1000),
            'ct_srv_src': np.random.randint(1, 100),
            'ct_state_ttl': np.random.randint(1, 50),
            'ct_dst_ltm': np.random.randint(1, 100),
            'ct_src_dport_ltm': np.random.randint(1, 100),
            'ct_dst_sport_ltm': np.random.randint(1, 100),
            'ct_dst_src_ltm': np.random.randint(1, 100),
            'is_ftp_login': np.random.choice([0, 1]),
            'ct_ftp_cmd': np.random.randint(0, 10),
            'ct_flw_http_mthd': np.random.randint(0, 10),
            'ct_src_ltm': np.random.randint(1, 100),
            'ct_srv_dst': np.random.randint(1, 100),
            'is_sm_ips_ports': np.random.choice([0, 1])
        }
    else:  # attack
        return {
            'dur': np.random.uniform(30, 300),
            'proto': np.random.choice([0, 1, 2]),
            'service': np.random.choice([0, 1, 2, 3]),
            'state': np.random.choice([0, 1, 2, 3]),
            'spkts': np.random.randint(100, 10000),
            'dpkts': np.random.randint(100, 10000),
            'sbytes': np.random.randint(1000, 100000),
            'dbytes': np.random.randint(1000, 100000),
            'rate': np.random.uniform(0.5, 1.0),
            'sttl': np.random.randint(1, 30),
            'dttl': np.random.randint(1, 30),
            'sload': np.random.uniform(1000, 10000),
            'dload': np.random.uniform(1000, 10000),
            'sloss': np.random.randint(10, 100),
            'dloss': np.random.randint(10, 100),
            'sinpkt': np.random.uniform(10, 100),
            'dinpkt': np.random.uniform(10, 100),
            'sjit': np.random.uniform(1, 10),
            'djit': np.random.uniform(1, 10),
            'swin': np.random.randint(0, 1024),
            'stcpb': np.random.randint(0, 1000),
            'dtcpb': np.random.randint(0, 1000),
            'dwin': np.random.randint(0, 1024),
            'tcprtt': np.random.uniform(10, 100),
            'synack': np.random.uniform(10, 100),
            'ackdat': np.random.uniform(10, 100),
            'smean': np.random.randint(100, 1000),
            'dmean': np.random.randint(100, 1000),
            'trans_depth': np.random.randint(5, 20),
            'response_body_len': np.random.randint(1000, 10000),
            'ct_srv_src': np.random.randint(100, 1000),
            'ct_state_ttl': np.random.randint(50, 200),
            'ct_dst_ltm': np.random.randint(100, 1000),
            'ct_src_dport_ltm': np.random.randint(100, 1000),
            'ct_dst_sport_ltm': np.random.randint(100, 1000),
            'ct_dst_src_ltm': np.random.randint(100, 1000),
            'is_ftp_login': np.random.choice([0, 1]),
            'ct_ftp_cmd': np.random.randint(10, 100),
            'ct_flw_http_mthd': np.random.randint(10, 100),
            'ct_src_ltm': np.random.randint(100, 1000),
            'ct_srv_dst': np.random.randint(100, 1000),
            'is_sm_ips_ports': np.random.choice([0, 1])
        }

def get_prediction_analytics():
    """
    Generate analytics from prediction history
    """
    predictions = load_prediction_log()
    
    if not predictions:
        return {
            'total_predictions': 0,
            'message': 'No prediction history available'
        }
    
    df = pd.DataFrame(predictions)
    
    # Basic statistics
    total_predictions = len(df)
    attack_count = sum(1 for verdict in df['final_verdict'] if verdict == 'Attack')
    normal_count = sum(1 for verdict in df['final_verdict'] if verdict == 'Normal')
    error_count = sum(1 for verdict in df['final_verdict'] if verdict == 'Error')
    
    # Model performance
    model_agreement = {}
    for pred in predictions:
        ensemble_verdict = pred['final_verdict']
        model_preds = pred.get('predictions', {})
        
        for model_name, model_pred in model_preds.items():
            model_verdict = "Attack" if model_pred == 1 else "Normal" if model_pred == 0 else "Error"
            agreement = 1 if model_verdict == ensemble_verdict else 0
            
            if model_name not in model_agreement:
                model_agreement[model_name] = []
            model_agreement[model_name].append(agreement)
    
    model_accuracy = {model: np.mean(agreements) for model, agreements in model_agreement.items()}
    
    # Time-based analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    hourly_attacks = df[df['final_verdict'] == 'Attack'].groupby('hour').size()
    
    analytics = {
        'total_predictions': total_predictions,
        'attack_count': attack_count,
        'normal_count': normal_count,
        'error_count': error_count,
        'attack_rate': attack_count / total_predictions,
        'model_accuracy': model_accuracy,
        'hourly_attack_pattern': hourly_attacks.to_dict(),
        'latest_prediction': predictions[-1] if predictions else None
    }
    
    return analytics

def export_predictions_to_csv(output_file="data/predictions/prediction_export.csv"):
    """
    Export prediction history to CSV for external analysis
    """
    predictions = load_prediction_log()
    
    if not predictions:
        print("❌ No predictions to export")
        return False
    
    try:
        # Flatten the prediction data for CSV export
        export_data = []
        for pred in predictions:
            row = {
                'timestamp': pred.get('timestamp'),
                'final_verdict': pred.get('final_verdict'),
                'ensemble_confidence': pred.get('ensemble_confidence', 0)
            }
            
            # Add individual model predictions
            for model_name, model_pred in pred.get('predictions', {}).items():
                row[f'{model_name}_prediction'] = model_pred
                row[f'{model_name}_confidence'] = pred.get('confidences', {}).get(model_name, 0)
            
            export_data.append(row)
        
        df_export = pd.DataFrame(export_data)
        ensure_directory_exists(output_file)
        df_export.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(export_data)} predictions to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting predictions: {e}")
        return False

def initialize_prediction_directories():
    """
    Initialize all required directories for real-time prediction system
    """
    directories = [
        "data/models/",
        "data/predictions/", 
        "data/processed/",
        "data/raw/",
        "logs/"
    ]
    
    for directory in directories:
        ensure_directory_exists(directory + "dummy_file.txt")
    
    print("✅ All prediction directories initialized")