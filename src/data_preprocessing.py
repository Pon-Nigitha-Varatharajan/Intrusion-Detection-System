import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
from config import settings

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def basic_preprocessing(self, df):
        """Step 1: Basic preprocessing as per paper"""
        if df is None or len(df) == 0:
            print("❌ Empty dataframe received for preprocessing")
            return df
            
        df_clean = df.copy()
        
        print(f"🔍 Before preprocessing: {df_clean.shape}")
        
        # 1. Handle missing values - only drop rows where ALL values are NaN
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(how='all')  # Only drop rows that are entirely NaN
        print(f"   After dropping fully NaN rows: {len(df_clean)} (removed {initial_rows - len(df_clean)})")
        
        # For columns with missing values, fill with appropriate values instead of dropping
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        if missing_cols:
            print(f"   Columns with missing values: {missing_cols}")
            for col in missing_cols:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna('Unknown', inplace=True)
        
        # 2. Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"   After removing duplicates: {len(df_clean)} (removed {initial_rows - len(df_clean)})")
        
        # 3. Remove spaces from column names
        df_clean.columns = [col.replace(' ', '').replace('-', '_') for col in df_clean.columns]
        print(f"   Cleaned column names")
        
        # 4. Optimize data types to reduce memory (only if we have data)
        if len(df_clean) > 0:
            for col in df_clean.select_dtypes(include=['int64']).columns:
                df_clean[col] = df_clean[col].astype('int32')
            
            for col in df_clean.select_dtypes(include=['float64']).columns:
                df_clean[col] = df_clean[col].astype('float32')
            print(f"   Optimized data types")
        
        print(f"✅ After preprocessing: {df_clean.shape}")
        return df_clean
    
    def feature_scaling(self, df):
        """Step 2: Feature scaling and encoding"""
        if df is None or len(df) == 0:
            print("❌ Empty dataframe received for feature scaling")
            return df
            
        df_scaled = df.copy()
        
        print(f"🔍 Before feature scaling: {df_scaled.shape}")
        
        # Identify numerical and categorical columns
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_scaled.select_dtypes(include=['object']).columns.tolist()
        
        print(f"   Numerical columns: {len(numerical_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
        
        # Remove target columns from scaling
        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        if 'attack_cat' in categorical_cols:
            categorical_cols.remove('attack_cat')
            
        print(f"   Numerical features to scale: {len(numerical_cols)}")
        print(f"   Categorical features to encode: {len(categorical_cols)}")
        
        # Standardize numerical features (only if we have numerical columns)
        if numerical_cols and len(df_scaled) > 0:
            try:
                df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
                print(f"✅ Standardized {len(numerical_cols)} numerical features")
            except Exception as e:
                print(f"❌ Error scaling numerical features: {e}")
                # If scaling fails, just use original values
                pass
        
        # Label encode categorical features (only if we have categorical columns)
        if categorical_cols and len(df_scaled) > 0:
            encoded_count = 0
            for col in categorical_cols:
                if col in df_scaled.columns and len(df_scaled[col].unique()) > 0:
                    df_scaled[col] = self.label_encoder.fit_transform(df_scaled[col].astype(str))
                    encoded_count += 1
            print(f"✅ Encoded {encoded_count} categorical features")
        
        print(f"✅ After feature scaling: {df_scaled.shape}")
        return df_scaled
    
    def random_oversampling(self, df):
        """Step 3: Simple Random Oversampling (manual implementation)"""
        if df is None or len(df) == 0 or 'label' not in df.columns:
            print("❌ Cannot perform oversampling - no data or no label column")
            return df
        
        # Count class distribution
        class_counts = df['label'].value_counts()
        print(f"🔍 Class distribution before oversampling: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            print("❌ Only one class found, cannot oversample")
            return df
            
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        # Separate majority and minority classes
        df_majority = df[df['label'] == majority_class]
        df_minority = df[df['label'] == minority_class]
        
        print(f"   Majority class ({majority_class}): {len(df_majority)} samples")
        print(f"   Minority class ({minority_class}): {len(df_minority)} samples")
        
        # Oversample minority class
        df_minority_oversampled = df_minority.sample(
            n=len(df_majority), 
            replace=True, 
            random_state=settings.RANDOM_STATE
        )
        
        # Combine majority class with oversampled minority class
        df_balanced = pd.concat([df_majority, df_minority_oversampled])
        
        balanced_counts = df_balanced['label'].value_counts()
        print(f"✅ After oversampling: {dict(balanced_counts)}")
        
        return df_balanced