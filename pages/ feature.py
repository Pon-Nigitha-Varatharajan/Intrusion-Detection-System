import streamlit as st
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
from src.utils import ensure_directory_exists
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

def main():
    st.title("🔧 Feature Engineering - SFE + PCA")
    st.write("""
    
    **Stacking Feature Embedded (SFE) + Principal Component Analysis (PCA)**
    
    - Uses **balanced dataset** from preprocessing (after Random Oversampling)
    - **SFE**: Add clustering results (K-Means + Gaussian Mixture) as meta-features
    - **PCA**: Reduce dimensionality to 10 most important features
    """)
    
    # Load the BALANCED data from preprocessing (Step 3 output - AFTER OVERSAMPLING)
    try:
        df_balanced = pd.read_csv("data/processed/balanced_data.csv")
        st.success(f"✅ Loaded balanced dataset: {df_balanced.shape}")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df_balanced):,}")
        with col2:
            st.metric("Features", f"{len(df_balanced.columns)}")
        with col3:
            if 'label' in df_balanced.columns:
                normal_count = len(df_balanced[df_balanced['label'] == 0])
                attack_count = len(df_balanced[df_balanced['label'] == 1])
                st.metric("Balance Ratio", f"{(attack_count/normal_count):.2f}")
            
    except FileNotFoundError:
        st.error("❌ Balanced dataset not found. Please run preprocessing first!")
        return
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Step 4: Stacking Feature Embedded
    st.subheader("🔧 Step 4: Stacking Feature Embedded (SFE)")
    st.write("""
    Adding clustering-based meta-features to balanced data:
    - **K-Means Clustering**: Groups similar network traffic patterns
    - **Gaussian Mixture Model**: Identifies probabilistic clusters
    - **Meta-features**: Cluster assignments are added as new features
    """)
    
    with st.spinner("Applying Stacking Feature Embedded..."):
        df_sfe = feature_engineer.stacking_feature_embedded(df_balanced)
    
    if df_sfe is not None and len(df_sfe) > 0:
        st.success(f"✅ SFE completed! Dataset shape: {df_sfe.shape}")
        
        # Show cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if 'kmeans_cluster' in df_sfe.columns:
                kmeans_counts = df_sfe['kmeans_cluster'].value_counts()
                fig_kmeans = px.pie(
                    values=kmeans_counts.values,
                    names=[f'Cluster {i}' for i in kmeans_counts.index],
                    title='K-Means Cluster Distribution'
                )
                st.plotly_chart(fig_kmeans, use_container_width=True)
                st.write(f"**K-Means Clusters:** {len(kmeans_counts)} clusters")
        
        with col2:
            if 'gmm_cluster' in df_sfe.columns:
                gmm_counts = df_sfe['gmm_cluster'].value_counts()
                fig_gmm = px.pie(
                    values=gmm_counts.values,
                    names=[f'Component {i}' for i in gmm_counts.index],
                    title='GMM Component Distribution'
                )
                st.plotly_chart(fig_gmm, use_container_width=True)
                st.write(f"**GMM Components:** {len(gmm_counts)} components")
    
    # Step 5: PCA
    st.subheader("📉 Step 5: Principal Component Analysis (PCA)")
    st.write(f"Reducing dimensionality from {len(df_sfe.columns)-2} features to {feature_engineer.pca.n_components} components...")
    
    with st.spinner("Applying PCA..."):
        df_pca = feature_engineer.apply_pca(df_sfe)
    
    if df_pca is not None and len(df_pca) > 0:
        st.success(f"✅ PCA completed! Final dataset shape: {df_pca.shape}")
        
        # Show PCA explained variance
        explained_variance = sum(feature_engineer.pca.explained_variance_ratio_)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Explained Variance", f"{explained_variance:.4f}")
        
        with col2:
            st.metric("PCA Components", f"{feature_engineer.pca.n_components}")
            
        with col3:
            if 'label' in df_pca.columns:
                normal_count = len(df_pca[df_pca['label'] == 0])
                attack_count = len(df_pca[df_pca['label'] == 1])
                st.metric("Records per Class", f"{normal_count:,} : {attack_count:,}")
        
        # Plot explained variance
        fig_var = px.bar(
            x=[f'PC{i+1}' for i in range(len(feature_engineer.pca.explained_variance_ratio_))],
            y=feature_engineer.pca.explained_variance_ratio_,
            title='PCA Explained Variance Ratio',
            labels={'x': 'Principal Components', 'y': 'Explained Variance Ratio'}
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Show individual component variances
        with st.expander("View Individual Component Variances"):
            for i, variance in enumerate(feature_engineer.pca.explained_variance_ratio_):
                st.write(f"- **PC{i+1}**: {variance:.4f} ({variance*100:.2f}%)")
    
    # Save FINAL engineered data
    st.subheader("💾 Save Final Training Data")
    if st.button("Save Final Training Dataset"):
        ensure_directory_exists("data/processed/")
        
        if 'df_sfe' in locals() and df_sfe is not None:
            df_sfe.to_csv("data/processed/sfe_balanced_data.csv", index=False)
            st.success("✅ SFE balanced data saved: `data/processed/sfe_balanced_data.csv`")
        
        if 'df_pca' in locals() and df_pca is not None:
            df_pca.to_csv("data/processed/final_training_data.csv", index=False)
            st.success("✅ FINAL training data saved: `data/processed/final_training_data.csv`")
            
            # Show final dataset info
            st.info("**🎯 Final Training Dataset Ready for Model Training:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **Records:** {len(df_pca):,}")
                st.write(f"- **Features:** {len(df_pca.columns)}")
                st.write(f"- **PCA Components:** {feature_engineer.pca.n_components}")
            with col2:
                st.write(f"- **Variance Explained:** {explained_variance:.4f}")
                if 'label' in df_pca.columns:
                    normal_count = len(df_pca[df_pca['label'] == 0])
                    attack_count = len(df_pca[df_pca['label'] == 1])
                    st.write(f"- **Class Balance:** {normal_count:,} : {attack_count:,}")
    
    # Display sample of final engineered data
    if 'df_pca' in locals() and df_pca is not None:
        st.subheader("🔍 Final Training Data Sample")
        st.dataframe(df_pca.head(10))
        
        st.balloons()
        st.success("🎉 Feature Engineering completed! Final training data is ready for model training.")

if __name__ == "__main__":
    main()