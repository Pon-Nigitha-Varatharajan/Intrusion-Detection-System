import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from config import settings

class FeatureEngineer:
    def __init__(self):
        self.kmeans = KMeans(
            n_clusters=settings.CLUSTERING_SETTINGS['n_clusters_kmeans'],
            random_state=settings.RANDOM_STATE
        )
        self.gmm = GaussianMixture(
            n_components=settings.CLUSTERING_SETTINGS['n_components_gmm'],
            random_state=settings.RANDOM_STATE
        )
        self.pca = PCA(n_components=settings.N_COMPONENTS_PCA)
    
    def stacking_feature_embedded(self, df):
        """Step 4: Stacking Feature Embedded (SFE) with clustering - AS PER PAPER"""
        print("🔧 Starting Stacking Feature Embedded (SFE) on BALANCED data as per paper...")
        
        if df is None or len(df) == 0:
            print("❌ Empty dataframe for SFE")
            return df
            
        # Remove ONLY target columns for clustering (keep all original features)
        feature_columns = [col for col in df.columns if col not in ['label', 'attack_cat']]
        X = df[feature_columns]
        
        print(f"   Features for clustering: {len(feature_columns)}")
        print(f"   Total records: {len(df)}")
        
        # Apply K-Means clustering
        print("   Applying K-Means clustering...")
        kmeans_clusters = self.kmeans.fit_predict(X)
        
        # Apply Gaussian Mixture clustering  
        print("   Applying Gaussian Mixture clustering...")
        gmm_clusters = self.gmm.fit_predict(X)
        
        # Create enhanced dataset: ORIGINAL FEATURES + CLUSTER META-FEATURES
        df_sfe = df.copy()
        df_sfe['kmeans_cluster'] = kmeans_clusters
        df_sfe['gmm_cluster'] = gmm_clusters
        
        print(f"✅ Added K-Means meta-feature ({self.kmeans.n_clusters} clusters)")
        print(f"✅ Added GMM meta-feature ({self.gmm.n_components} components)")
        print(f"   Enhanced dataset shape: {df_sfe.shape}")
        print(f"   Total features after SFE: {len(df_sfe.columns)}")
        
        return df_sfe
    
    def apply_pca(self, df):
        """Step 5: Apply PCA on SFE-enhanced BALANCED data - AS PER PAPER"""
        print("🔧 Applying PCA on SFE-enhanced BALANCED data as per paper...")
        
        if df is None or len(df) == 0:
            print("❌ Empty dataframe for PCA")
            return df
            
        # Remove ONLY target columns for PCA (keep all features including clusters)
        feature_columns = [col for col in df.columns if col not in ['label', 'attack_cat']]
        X = df[feature_columns]
        
        print(f"   Features before PCA (including SFE meta-features): {len(feature_columns)}")
        print(f"   Total records: {len(df)}")
        
        # Apply PCA on the ENHANCED feature set (original + clusters)
        X_pca = self.pca.fit_transform(X)
        
        # Create new dataframe with PCA features
        pca_columns = [f'PC{i+1}' for i in range(settings.N_COMPONENTS_PCA)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)
        
        # Add back target columns ONLY
        if 'label' in df.columns:
            df_pca['label'] = df['label'].values
        if 'attack_cat' in df.columns:
            df_pca['attack_cat'] = df['attack_cat'].values
        
        print(f"✅ PCA reduced {len(feature_columns)} features to {settings.N_COMPONENTS_PCA} components")
        print(f"   Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        print(f"   Final dataset shape: {df_pca.shape}")
        
        return df_pca