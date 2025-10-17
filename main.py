import streamlit as st

st.set_page_config(
    page_title="UNSW-NB15 IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛡️ UNSW-NB15 Network Intrusion Detection System")
    st.markdown("---")
    
    st.subheader("Welcome to the Intrusion Detection System")
    st.write("""
    This application implements a machine learning-based Network Intrusion Detection System 
    using the UNSW-NB15 dataset. The system follows the methodology from the research paper:
    
    **"Machine learning-based network intrusion detection for big and imbalanced data using 
    oversampling, stacking feature embedding and feature extraction"**
    """)
    
    st.info("""
    ### Key Features:
    - **Data Exploration**: Analyze the UNSW-NB15 dataset
    - **Preprocessing**: Handle missing values, normalization, and oversampling
    - **Feature Engineering**: Stacking Feature Embedded (SFE) with PCA
    - **Model Training**: Train DT, RF, ET, and XGB models
    - **Evaluation**: Performance metrics and visualization
    - **Real-time Prediction**: Test new network traffic data
    """)
    
    # Project Structure
    st.subheader("📁 Project Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core Modules:**")
        st.write("• `src/data_preprocessing.py` - Data cleaning & normalization")
        st.write("• `src/feature_engineering.py` - SFE & PCA implementation")
        st.write("• `src/model_training.py` - ML model training")
        st.write("• `src/model_evaluation.py` - Performance metrics")
        st.write("• `src/utils.py` - Helper functions")
    
    with col2:
        st.write("**Application Pages:**")
        st.write("• 🏠 Home - Project overview")
        st.write("• 📊 Data Exploration - Dataset analysis")
        st.write("• ⚙️ Preprocessing - Data preparation")
        st.write("• 🤖 Model Training - Train ML models")
        st.write("• 📈 Evaluation - Model performance")
        st.write("• 🔍 Real-time Prediction - Test new data")
    
    # Paper Methodology
    st.subheader("📚 Research Paper Methodology")
    st.write("""
    This implementation follows the exact methodology from the paper:
    
    1. **Data Preprocessing**: Handle missing values, remove duplicates, optimize data types
    2. **Feature Scaling**: Standardization and label encoding
    3. **Random Oversampling (RO)**: Address class imbalance
    4. **Stacking Feature Embedded (SFE)**: Add clustering-based meta-features
    5. **Principal Component Analysis (PCA)**: Reduce dimensionality to 10 features
    6. **Model Training**: Decision Tree, Random Forest, Extra Trees, XGBoost
    7. **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC curves
    """)
    
    # Quick Start Guide
    st.subheader("🚀 Quick Start Guide")
    st.write("""
    1. **Start with Data Exploration** - Understand your dataset
    2. **Run Preprocessing** - Clean and prepare the data
    3. **Train Models** - Build machine learning classifiers
    4. **Evaluate Performance** - Analyze model results
    5. **Make Predictions** - Test on new network traffic
    """)
    
    st.warning("""
    ⚠️ **Important**: Make sure your UNSW-NB15 dataset files are placed in the `data/raw/` folder 
    before starting the preprocessing steps!
    """)
    
    # Dataset Information
    st.subheader("📊 Dataset Information")
    st.write("""
    **UNSW-NB15 Dataset Features:**
    - Contains modern normal and attack activities
    - 9 types of attacks: Fuzzers, DoS, Exploits, Reconnaissance, etc.
    - 49 features including network flow statistics
    - Both binary (normal/attack) and multi-class classification
    - Realistic network traffic patterns
    """)

if __name__ == "__main__":
    main()