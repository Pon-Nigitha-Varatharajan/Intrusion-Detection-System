import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.utils import load_combined_dataset, ensure_directory_exists
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    st.title("⚙️ Data Preprocessing")
    st.write("""
    This page implements the preprocessing steps from the research paper:
    - Handling missing values
    - Removing duplicates
    - Data type optimization
    - Feature scaling and encoding
    - Random Oversampling (RO) for class imbalance
    """)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    if st.button("Load and Preprocess Data"):
        with st.spinner("Loading dataset..."):
            df = load_combined_dataset()
        
        if df is None:
            st.error("Please place UNSW-NB15 dataset files in data/raw/ folder")
            return
        
        st.success(f"Dataset loaded! Shape: {df.shape}")
        
        # Display original data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Records", f"{len(df):,}")
        with col2:
            st.metric("Original Features", f"{len(df.columns)}")
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display first few rows
        st.subheader("📋 Original Data Sample")
        st.dataframe(df.head())
        
        # Step 1: Basic Preprocessing
        st.subheader("🔄 Step 1: Basic Preprocessing")
        with st.spinner("Performing basic preprocessing..."):
            df_processed = preprocessor.basic_preprocessing(df)
        
        # ✅ AUTO-SAVE after Step 1
        if df_processed is not None and len(df_processed) > 0:
            ensure_directory_exists("data/processed/")
            df_processed.to_csv("data/processed/preprocessed_data.csv", index=False)
            st.success(f"✅ Step 1 completed! Saved {len(df_processed):,} records to 'data/processed/preprocessed_data.csv'")
        else:
            st.error("❌ Step 1 failed - No data after preprocessing")
            return
        
        st.write("**Basic Preprocessing Completed:**")
        st.write("- ✅ Handled missing values")
        st.write("- ✅ Removed duplicate rows") 
        st.write("- ✅ Optimized data types")
        st.write("- ✅ Cleaned column names")
        
        # Step 2: Feature Scaling
        st.subheader("📊 Step 2: Feature Scaling & Encoding")
        with st.spinner("Performing feature scaling..."):
            df_scaled = preprocessor.feature_scaling(df_processed)
        
        # ✅ AUTO-SAVE after Step 2
        if df_scaled is not None and len(df_scaled) > 0:
            df_scaled.to_csv("data/processed/normalized_data.csv", index=False)
            st.success(f"✅ Step 2 completed! Saved {len(df_scaled):,} records to 'data/processed/normalized_data.csv'")
        else:
            st.error("❌ Step 2 failed - No data after feature scaling")
            return
        
        st.write("**Feature Scaling Completed:**")
        st.write("- ✅ Standardized numerical features")
        st.write("- ✅ Label encoded categorical features")
        
        # Step 3: Random Oversampling (only if label exists)
        if 'label' in df_scaled.columns:
            st.subheader("⚖️ Step 3: Random Oversampling (RO)")
            
            original_counts = df_scaled['label'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Oversampling:**")
                st.write(f"Class 0 (Normal): {original_counts.get(0, 0):,}")
                st.write(f"Class 1 (Attack): {original_counts.get(1, 0):,}")
                if 0 in original_counts and 1 in original_counts:
                    st.write(f"Imbalance Ratio: {original_counts[1]/original_counts[0]:.2f}")
            
            with st.spinner("Applying Random Oversampling..."):
                df_balanced = preprocessor.random_oversampling(df_scaled)
            
            # ✅ AUTO-SAVE after Step 3
            if df_balanced is not None and len(df_balanced) > 0:
                df_balanced.to_csv("data/processed/balanced_data.csv", index=False)
                st.success(f"✅ Step 3 completed! Saved {len(df_balanced):,} records to 'data/processed/balanced_data.csv'")
                
                balanced_counts = df_balanced['label'].value_counts()
                with col2:
                    st.write("**After Oversampling:**")
                    st.write(f"Class 0 (Normal): {balanced_counts.get(0, 0):,}")
                    st.write(f"Class 1 (Attack): {balanced_counts.get(1, 0):,}")
                    st.write("✅ Classes balanced!")
            else:
                st.warning("❌ Step 3 failed - No data after oversampling")
        else:
            st.warning("No 'label' column found for oversampling")
            df_balanced = df_scaled
        
        # Display sample of processed data
        st.subheader("🔍 Processed Data Sample")
        st.dataframe(df_scaled.head(10))
        
        # ✅ Show file locations
        st.subheader("💾 Saved File Locations")
        file_locations = {
            "Preprocessed Data": "data/processed/preprocessed_data.csv",
            "Normalized Data": "data/processed/normalized_data.csv", 
            "Balanced Data": "data/processed/balanced_data.csv"
        }
        
        for file_desc, file_path in file_locations.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                st.success(f"✅ **{file_desc}**: `{file_path}` ({file_size:.2f} MB)")
            else:
                st.error(f"❌ **{file_desc}**: `{file_path}` (Not found)")
        
        st.balloons()

if __name__ == "__main__":
    main()