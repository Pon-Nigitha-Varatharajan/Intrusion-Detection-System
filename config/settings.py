# Dataset paths
# Dataset paths - Using ONLY training and testing sets as per paper (257,673 total records)
DATA_RAW_PATH = "data/raw/"
DATA_TRAINING_PATH = "data/raw/UNSW_NB15_training-set.csv"    # 175,341 records
DATA_TESTING_PATH = "data/raw/UNSW_NB15_testing-set.csv"      # 82,332 records
# Total: 257,673 records (matches paper specification)

DATA_PROCESSED_PATH = "data/processed/preprocessed_data.csv"
DATA_NORMALIZED_PATH = "data/processed/normalized_data.csv"
DATA_BALANCED_PATH = "data/processed/balanced_data.csv"

# Model paths
MODELS_PATH = "data/models/"

# Preprocessing settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_COMPONENTS_PCA = 10
CV_FOLDS = 10  # Add this line


# Model parameters
N_ESTIMATORS = 100
MAX_DEPTH = 10

# UNSW-NB15 specific settings - Based on actual dataset structure
UNSW_FEATURES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 
    'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
    'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 
    'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Attack categories mapping (based on UNSW-NB15 dataset)
ATTACK_CATEGORIES = {
    'Normal': 'Normal',
    'Generic': 'Generic',
    'Exploits': 'Exploits',
    'Fuzzers': 'Fuzzers',
    'DoS': 'DoS',
    'Reconnaissance': 'Reconnaissance',
    'Analysis': 'Analysis',
    'Backdoor': 'Backdoor',
    'Shellcode': 'Shellcode',
    'Worms': 'Worms'
}

# Binary classification mapping
BINARY_LABELS = {
    'Normal': 0,
    'Attack': 1
}

# Feature groups for analysis
NUMERICAL_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 
    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 
    'synack', 'ackdat'
]

CATEGORICAL_FEATURES = [
    'proto', 'state', 'service'
]

# Clustering settings for SFE (Stacking Feature Embedded)
CLUSTERING_SETTINGS = {
    'n_clusters_kmeans': 5,
    'n_components_gmm': 3,
    'random_state': 42
}

# Model names for reference
MODEL_NAMES = {
    'dt': 'Decision Tree',
    'rf': 'Random Forest', 
    'et': 'Extra Trees',
    'xgb': 'XGBoost'
}

# Performance metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']