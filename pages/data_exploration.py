import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.utils import load_combined_dataset, ensure_directory_exists
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def main():
    st.title("🔍 Data Exploration")
    st.write("""
    This page provides comprehensive exploratory data analysis (EDA) for the UNSW-NB15 dataset.
    Explore the dataset's characteristics, distributions, and patterns before modeling.
    """)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data section
    st.header("📂 Data Loading")
    
    if st.button("Load Dataset for Exploration"):
        with st.spinner("Loading dataset..."):
            df = load_combined_dataset()
        
        if df is None:
            st.error("Please place UNSW-NB15 dataset files in data/raw/ folder")
            return
        
        # Store in session state
        st.session_state.df = df
        st.session_state.exploration_done = True
        
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    
    # Check if data is loaded
    if 'df' not in st.session_state:
        st.info("👆 Click the button above to load the dataset for exploration")
        return
    
    df = st.session_state.df
    
    # Basic Dataset Information
    st.header("📊 Basic Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns)}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        missing_total = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_total:,}")
    
    # Display first few rows
    st.subheader("📋 Data Sample")
    sample_size = st.slider("Sample size to display", 5, 50, 10)
    st.dataframe(df.head(sample_size))
    
    # Data Types Information
    st.subheader("🔧 Data Types")
    dtype_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df)) * 100
    })
    st.dataframe(dtype_info)
    
    # Missing Values Analysis
    st.header("🎯 Missing Values Analysis")
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values by Column**")
            st.dataframe(missing_df)
        
        with col2:
            fig = px.bar(missing_df.head(10), x='Column', y='Missing Percentage',
                        title='Top 10 Columns with Missing Values',
                        color='Missing Percentage')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Target Variable Analysis
    st.header("🎯 Target Variable Analysis")
    
    if 'label' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            class_dist = df['label'].value_counts()
            fig_class = px.pie(values=class_dist.values, names=class_dist.index,
                             title='Class Distribution (Normal vs Attack)',
                             color=class_dist.index,
                             color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig_class, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(x=class_dist.index, y=class_dist.values,
                           labels={'x': 'Class', 'y': 'Count'},
                           title='Class Distribution Count',
                           color=class_dist.index,
                           color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write(f"**Class Distribution:**")
        st.write(f"- Class 0 (Normal): {class_dist.get(0, 0):,} samples ({class_dist.get(0, 0)/len(df)*100:.2f}%)")
        st.write(f"- Class 1 (Attack): {class_dist.get(1, 0):,} samples ({class_dist.get(1, 0)/len(df)*100:.2f}%)")
        
        if 0 in class_dist and 1 in class_dist:
            imbalance_ratio = max(class_dist[0], class_dist[1]) / min(class_dist[0], class_dist[1])
            st.write(f"**Imbalance Ratio:** {imbalance_ratio:.2f}")
    
    # Attack Category Analysis
    if 'attack_cat' in df.columns:
        st.subheader("🦠 Attack Category Distribution")
        
        attack_dist = df['attack_cat'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_attack = px.bar(x=attack_dist.index, y=attack_dist.values,
                              title='Attack Categories Distribution',
                              labels={'x': 'Attack Category', 'y': 'Count'})
            fig_attack.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_attack, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(values=attack_dist.values, names=attack_dist.index,
                           title='Attack Categories Proportion')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Numerical Features Analysis
    st.header("📈 Numerical Features Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        st.write(f"**Found {len(numerical_cols)} numerical features**")
        
        # Select numerical features to explore
        selected_num_cols = st.multiselect(
            "Select numerical features to explore:",
            numerical_cols,
            default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
        )
        
        if selected_num_cols:
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            st.dataframe(df[selected_num_cols].describe())
            
            # Distribution plots
            st.subheader("📊 Distribution Plots")
            
            cols_per_row = 2
            rows = (len(selected_num_cols) + cols_per_row - 1) // cols_per_row
            
            for i in range(0, len(selected_num_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(selected_num_cols):
                        feature = selected_num_cols[i + j]
                        with col:
                            fig = px.histogram(df, x=feature, 
                                             title=f'Distribution of {feature}',
                                             marginal='box')
                            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Features Analysis
    st.header("📊 Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write(f"**Found {len(categorical_cols)} categorical features**")
        
        # Remove target columns from categorical analysis
        if 'attack_cat' in categorical_cols:
            categorical_cols.remove('attack_cat')
        
        if categorical_cols:
            selected_cat_col = st.selectbox(
                "Select categorical feature to explore:",
                categorical_cols
            )
            
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(x=value_counts.index.astype(str), y=value_counts.values,
                               title=f'Distribution of {selected_cat_col}',
                               labels={'x': selected_cat_col, 'y': 'Count'})
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show value counts table
                    st.write(f"**Value Counts for {selected_cat_col}**")
                    st.dataframe(value_counts.head(10))
                
                st.write(f"**Unique values in {selected_cat_col}:** {df[selected_cat_col].nunique()}")
    
    # Correlation Analysis
    st.header("🔗 Correlation Analysis")
    
    if len(numerical_cols) > 1:
        # Select numerical features for correlation (limit to avoid performance issues)
        corr_cols = st.multiselect(
            "Select features for correlation analysis:",
            numerical_cols,
            default=numerical_cols[:10] if len(numerical_cols) >= 10 else numerical_cols
        )
        
        if len(corr_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[corr_cols].corr()
            
            # Correlation heatmap
            fig = px.imshow(corr_matrix,
                          title='Feature Correlation Heatmap',
                          color_continuous_scale='RdBu_r',
                          aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Find highly correlated features
            st.subheader("📌 Highly Correlated Features")
            high_corr_threshold = st.slider("Correlation threshold", 0.7, 0.95, 0.8)
            
            # Create matrix of high correlations
            high_corr_matrix = corr_matrix.abs()
            np.fill_diagonal(high_corr_matrix.values, 0)  # Ignore diagonal
            
            high_corr_pairs = []
            for i in range(len(corr_cols)):
                for j in range(i+1, len(corr_cols)):
                    if high_corr_matrix.iloc[i, j] > high_corr_threshold:
                        high_corr_pairs.append({
                            'Feature 1': corr_cols[i],
                            'Feature 2': corr_cols[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
                st.write(f"**Found {len(high_corr_pairs)} feature pairs with correlation > {high_corr_threshold}**")
                st.dataframe(high_corr_df)
            else:
                st.info(f"No feature pairs found with correlation > {high_corr_threshold}")
    
    # Outlier Detection
    st.header("📊 Outlier Detection")
    
    if numerical_cols:
        selected_outlier_col = st.selectbox(
            "Select numerical feature for outlier detection:",
            numerical_cols
        )
        
        if selected_outlier_col:
            # Calculate outliers using IQR method
            Q1 = df[selected_outlier_col].quantile(0.25)
            Q3 = df[selected_outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_outlier_col] < lower_bound) | 
                         (df[selected_outlier_col] > upper_bound)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = px.box(df, y=selected_outlier_col, 
                           title=f'Box Plot of {selected_outlier_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write(f"**Outlier Statistics for {selected_outlier_col}**")
                st.write(f"- Q1 (25th percentile): {Q1:.2f}")
                st.write(f"- Q3 (75th percentile): {Q3:.2f}")
                st.write(f"- IQR: {IQR:.2f}")
                st.write(f"- Lower bound: {lower_bound:.2f}")
                st.write(f"- Upper bound: {upper_bound:.2f}")
                st.write(f"- **Outliers detected:** {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    
    # Data Quality Report
    st.header("📋 Data Quality Report")
    
    if st.button("Generate Comprehensive Data Quality Report"):
        with st.spinner("Generating data quality report..."):
            # Basic quality metrics
            quality_metrics = {
                'Total Records': len(df),
                'Total Features': len(df.columns),
                'Missing Values': df.isnull().sum().sum(),
                'Duplicate Rows': df.duplicated().sum(),
                'Memory Usage (MB)': df.memory_usage(deep=True).sum() / (1024**2),
                'Numerical Features': len(df.select_dtypes(include=[np.number]).columns),
                'Categorical Features': len(df.select_dtypes(include=['object']).columns)
            }
            
            # Display quality metrics
            quality_df = pd.DataFrame(list(quality_metrics.items()), 
                                    columns=['Metric', 'Value'])
            st.dataframe(quality_df)
            
            # Feature completeness
            completeness = (df.count() / len(df)) * 100
            completeness_df = pd.DataFrame({
                'Feature': completeness.index,
                'Completeness (%)': completeness.values
            }).sort_values('Completeness (%)')
            
            st.subheader("Feature Completeness")
            st.dataframe(completeness_df)
    
    # Export Exploration Results
    st.header("💾 Export Exploration Results")
    
    if st.button("Save Exploration Summary"):
        ensure_directory_exists("reports/")
        
        # Generate and save basic exploration summary
        exploration_summary = {
            'dataset_shape': df.shape,
            'missing_values_total': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'numerical_features_count': len(numerical_cols),
            'categorical_features_count': len(categorical_cols)
        }
        
        if 'label' in df.columns:
            exploration_summary['class_distribution'] = df['label'].value_counts().to_dict()
        
        # Save to file
        summary_df = pd.DataFrame(list(exploration_summary.items()), 
                                columns=['Metric', 'Value'])
        summary_df.to_csv("reports/exploration_summary.csv", index=False)
        
        st.success("✅ Exploration summary saved to 'reports/exploration_summary.csv'")
    
    st.balloons()

if __name__ == "__main__":
    main()