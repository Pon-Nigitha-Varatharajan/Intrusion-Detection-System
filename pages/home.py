import streamlit as st

def app():
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("---")
    
    st.subheader("About this Project")
    st.write("""
    This application uses machine learning to detect network intrusions using the UNSW-NB15 dataset.
    
    **Features:**
    - Data Exploration and Visualization
    - Data Preprocessing and Feature Engineering
    - Machine Learning Model Training
    - Model Evaluation and Performance Metrics
    - Real-time Network Traffic Prediction
    """)
    
    st.subheader("Dataset Information")
    st.write("""
    **UNSW-NB15 Dataset:**
    - Comprehensive network intrusion detection dataset
    - Contains modern attack behaviors
    - 49 features with normal and attack traffic
    - Multiple attack categories
    """)
    
    st.info("Use the sidebar to navigate through different sections of the application.")

if __name__ == "__main__":
    app()