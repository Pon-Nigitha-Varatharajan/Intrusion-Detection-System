# pages/5_🤖_Model_Training.py

import streamlit as st
import pandas as pd
import numpy as np
from src.model_training import ModelTrainer
from src.utils import ensure_directory_exists
import plotly.express as px
import plotly.graph_objects as go
import os

def main():
    st.title("🤖 Model Training")
    
    st.markdown("""
    ### Step 7: ML Model Training with K-Fold Cross-Validation
    
    Training 4 machine learning models as per the research paper:
    - **Decision Tree (DT)**
    - **Random Forest (RF)**
    - **Extra Trees (ET)**
    - **XGBoost (XGB)**
    
    Using **10-Fold Cross-Validation** for robust performance evaluation.
    """)
    
    # Check if final training data exists
    data_path = "data/processed/final_training_data.csv"
    if not os.path.exists(data_path):
        st.error("❌ Final training data not found!")
        st.info("👉 Please complete **Feature Engineering** first")
        return
    
    # Initialize trainer
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ModelTrainer()
    
    trainer = st.session_state.trainer
    
    # Load data
    if trainer.X_train is None:
        with st.spinner("Loading training data..."):
            if not trainer.load_data(data_path):
                st.error("❌ Failed to load data!")
                return
        
        st.success("✅ Training data loaded successfully!")
        
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", f"{len(trainer.X_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(trainer.X_test):,}")
        with col3:
            st.metric("Features (PCA)", len(trainer.X_train.columns))
        with col4:
            train_ratio = len(trainer.X_train) / (len(trainer.X_train) + len(trainer.X_test))
            st.metric("Train/Test Split", f"{train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")
    
    st.markdown("---")
    
    # Training options
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        use_cv = st.checkbox("Use 10-Fold Cross-Validation", value=True, 
                            help="As per paper methodology")
    with col2:
        train_all = st.checkbox("Train All Models", value=True,
                               help="Train all 4 models at once")
    
    # Model selection for individual training
    if not train_all:
        model_to_train = st.selectbox(
            "Select Model to Train",
            options=list(trainer.models.keys())
        )
    
    st.markdown("---")
    
    # Training buttons
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if train_all:
            train_button = st.button("🚀 Train All Models", type="primary", use_container_width=True)
        else:
            train_button = st.button(f"🚀 Train {model_to_train}", type="primary", use_container_width=True)
    
    with col2:
        load_button = st.button("📂 Load Saved Models", type="secondary", use_container_width=True)
    
    # Training
    if train_button:
        with st.spinner("Training models... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if train_all:
                status_text.text("Training all models...")
                results = trainer.train_all_models(use_cross_validation=use_cv)
                progress_bar.progress(100)
            else:
                status_text.text(f"Training {model_to_train}...")
                results = trainer.train_model(model_to_train, use_cross_validation=use_cv)
                progress_bar.progress(100)
        
        st.success("✅ Training completed!")
        st.balloons()
        
        # Save models automatically
        with st.spinner("Saving models..."):
            ensure_directory_exists("data/models/")
            trainer.save_all_models()
        
        st.session_state.training_complete = True
    
    # Load saved models
    if load_button:
        with st.spinner("Loading saved models..."):
            models_dir = "data/models/"
            if os.path.exists(models_dir):
                loaded_count = 0
                for model_name in trainer.models.keys():
                    if trainer.load_model(model_name):
                        loaded_count += 1
                
                if loaded_count > 0:
                    st.success(f"✅ Loaded {loaded_count} saved models")
                    st.session_state.training_complete = True
                else:
                    st.warning("No saved models found")
            else:
                st.error("Models directory not found")
    
    # Display results if training is complete
    if hasattr(st.session_state, 'training_complete') and st.session_state.training_complete:
        st.markdown("---")
        display_training_results(trainer)

def display_training_results(trainer):
    """Display training results and comparisons"""
    st.header("📊 Training Results")
    
    if not trainer.training_results:
        st.info("No training results available yet")
        return
    
    # Overall performance comparison
    st.subheader("🏆 Model Performance Comparison")
    
    # Get comparison dataframe
    comparison_df = trainer.get_comparison_dataframe()
    
    # Display metrics table
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}', 
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'ROC AUC': '{:.4f}',
                'Training Time (s)': '{:.2f}',
                'CV Mean': '{:.4f}',
                'CV Std': '{:.4f}'
            }),
            use_container_width=True
        )
    
    with col2:
        # Find best model
        best_row = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        st.metric("Best Model", best_row['Model'])
        st.metric("Best Accuracy", f"{best_row['Accuracy']:.4f}")
        st.metric("Training Time", f"{best_row['Training Time (s)']:.2f}s")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📈 Performance Visualizations")
    
    # Metric comparison chart
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(4),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training time comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Training time bar chart
        fig_time = px.bar(
            comparison_df,
            x='Model',
            y='Training Time (s)',
            title='Training Time Comparison',
            color='Training Time (s)',
            color_continuous_scale='Viridis'
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Cross-validation results
        if 'CV Mean' in comparison_df.columns and not comparison_df['CV Mean'].isna().all():
            fig_cv = go.Figure()
            
            fig_cv.add_trace(go.Bar(
                name='CV Mean Accuracy',
                x=comparison_df['Model'],
                y=comparison_df['CV Mean'],
                error_y=dict(
                    type='data',
                    array=comparison_df['CV Std'],
                    visible=True
                ),
                text=comparison_df['CV Mean'].round(4),
                textposition='auto'
            ))
            
            fig_cv.update_layout(
                title='Cross-Validation Results (10-Fold)',
                yaxis_title='Accuracy',
                yaxis_range=[0, 1.05],
                height=400
            )
            st.plotly_chart(fig_cv, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed model results
    st.subheader("🔍 Detailed Model Analysis")
    
    # Model selector for detailed view
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        options=list(trainer.training_results.keys())
    )
    
    if selected_model:
        display_model_details(trainer, selected_model)
    
    st.markdown("---")
    
    # Model saving options
    st.subheader("💾 Model Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save All Models", use_container_width=True):
            with st.spinner("Saving models..."):
                ensure_directory_exists("data/models/")
                trainer.save_all_models()
                st.success("✅ All models saved successfully!")
    
    with col2:
        if st.button("🔄 Retrain All Models", use_container_width=True):
            st.session_state.training_complete = False
            st.rerun()
    
    with col3:
        # Download comparison results
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="model_training_results.csv",
            mime="text/csv",
            use_container_width=True
        )

def display_model_details(trainer, model_name):
    """Display detailed results for a specific model"""
    results = trainer.training_results[model_name]
    
    st.markdown(f"### {model_name} - Detailed Results")
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{results['test_accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{results['test_precision']:.4f}")
    with col3:
        st.metric("Recall", f"{results['test_recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{results['test_f1']:.4f}")
    with col5:
        if 'test_roc_auc' in results:
            st.metric("ROC AUC", f"{results['test_roc_auc']:.4f}")
        else:
            st.metric("Training Time", f"{results['training_time']:.2f}s")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        
        # Create confusion matrix visualization
        cm = results['test_confusion_matrix']
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Attack'],
            y=['Actual Normal', 'Actual Attack'],
            text=[[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"],
                  [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]],
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        fig_cm.update_layout(
            title=f'Test Set Confusion Matrix',
            width=400,
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification Report
        st.subheader("Classification Report")
        
        y_true = trainer.y_test
        y_pred = results['y_test_pred']
        
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(
            report_df.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1-score': '{:.4f}',
                'support': '{:.0f}'
            }),
            use_container_width=True
        )
    
    # Cross-validation results if available
    if results.get('cv_scores') is not None:
        st.subheader("📊 Cross-Validation Results")
        
        cv_scores = results['cv_scores']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CV Mean Accuracy", f"{results['cv_mean']:.4f}")
        with col2:
            st.metric("CV Std", f"{results['cv_std']:.4f}")
        with col3:
            st.metric("CV Range", f"{cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        # CV scores distribution
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker_color='lightblue'
        ))
        fig_cv.add_hline(y=results['cv_mean'], line_dash="dash", line_color="red",
                        annotation_text=f"Mean: {results['cv_mean']:.4f}")
        
        fig_cv.update_layout(
            title="Cross-Validation Scores per Fold",
            xaxis_title="Fold",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1.05],
            height=400
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # ROC Curve if available
    if 'roc_curve' in results and results['roc_curve'] is not None:
        st.subheader("📈 ROC Curve")
        
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        auc_score = results.get('test_roc_auc', 0.0)
        
        fig_roc = go.Figure()
        
        # ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc_score:.4f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random classifier',
            line=dict(dash='dash', color='red')
        ))
        
        fig_roc.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)

if __name__ == "__main__":
    main()