import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from src.real_time_predictor import RealTimePredictor
from src.utils import ensure_directory_exists

def main():
    st.title("🔮 Real-Time Intrusion Detection")
    
    st.markdown("""
    ### Real-Time Network Traffic Analysis
    
    Upload network traffic data or simulate real-time traffic for intrusion detection.
    The system uses trained ML models to detect potential attacks in real-time.
    """)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = RealTimePredictor()
        st.session_state.prediction_log = []
        st.session_state.auto_refresh = False
    
    predictor = st.session_state.predictor
    
    # Check if models are loaded
    if not predictor.models_loaded:
        st.warning("⚠️ Models not loaded. Loading models...")
        
        models_dir = "data/models/"
        if os.path.exists(models_dir):
            with st.spinner("Loading trained models..."):
                if predictor.load_models(models_dir):
                    st.success(f"✅ Loaded {len(predictor.models)} models successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load models!")
                    st.info("👉 Please complete **Model Training** first")
                    return
        else:
            st.error("❌ Models directory not found!")
            st.info("👉 Please complete **Model Training** first")
            return
    
    # Display loaded models
    with st.expander("📊 Loaded Models", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Loaded", len(predictor.models))
        with col2:
            st.metric("Features", len(predictor.feature_columns) if predictor.feature_columns else "N/A")
        with col3:
            # Safe way to get PCA components
            pca_components = "N/A"
            if predictor.pca is not None:
                try:
                    if hasattr(predictor.pca, 'n_components_'):
                        pca_components = predictor.pca.n_components_
                    elif hasattr(predictor.pca, 'n_components'):
                        pca_components = predictor.pca.n_components
                except:
                    pca_components = "Unknown"
            st.metric("PCA Components", pca_components)
        
        st.write("**Available Models:**")
        for model_name in predictor.models.keys():
            st.write(f"✓ {model_name}")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Single Prediction", 
        "📊 Batch Prediction", 
        "🔄 Live Monitoring",
        "📈 Statistics"
    ])
    
    # Tab 1: Single Prediction
    with tab1:
        single_prediction_tab(predictor)
    
    # Tab 2: Batch Prediction
    with tab2:
        batch_prediction_tab(predictor)
    
    # Tab 3: Live Monitoring
    with tab3:
        live_monitoring_tab(predictor)
    
    # Tab 4: Statistics
    with tab4:
        statistics_tab(predictor)

def single_prediction_tab(predictor):
    """Single instance prediction interface"""
    st.subheader("🎯 Single Instance Prediction")
    
    st.markdown("Upload a CSV file with a single network flow or enter values manually.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with network traffic data",
            type=['csv'],
            key='single_upload'
        )
    
    with col2:
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            options=['Ensemble (All Models)'] + list(predictor.models.keys()),
            key='single_model'
        )
        
        if model_choice == 'Ensemble (All Models)':
            voting_method = st.radio(
                "Voting Method",
                options=['hard', 'soft'],
                help="Hard: Majority vote, Soft: Average probabilities"
            )
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded file with {len(df)} row(s) and {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            # Predict button
            if st.button("🔮 Make Prediction", type="primary", width='stretch'):
                with st.spinner("Making prediction..."):
                    # Take first row for single prediction
                    single_row = df.iloc[[0]]
                    
                    # Make prediction
                    if model_choice == 'Ensemble (All Models)':
                        result = predictor.predict_ensemble(single_row, voting=voting_method)
                    else:
                        result = predictor.predict_single(single_row, model_name=model_choice)
                    
                    # Check for errors
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        return
                    
                    # Add to history
                    predictor.add_to_history(result, single_row)
                    st.session_state.prediction_log.append(result)
                    
                    # Display result
                    display_prediction_result(result, single_row)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("💡 Upload a CSV file to make predictions")
        
        # Example format
        with st.expander("📄 Expected CSV Format"):
            st.markdown("""
            Your CSV should contain network flow features. Example columns:
            - Flow duration
            - Total Fwd Packets
            - Total Backward Packets
            - Flow Bytes/s
            - Flow Packets/s
            - Protocol
            - And other network features...
            
            The system will automatically handle feature engineering and preprocessing.
            """)

def batch_prediction_tab(predictor):
    """Batch prediction interface"""
    st.subheader("📊 Batch Prediction")
    
    st.markdown("Upload a CSV file with multiple network flows for batch prediction.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with network traffic data",
            type=['csv'],
            key='batch_upload'
        )
    
    with col2:
        model_choice = st.selectbox(
            "Select Model",
            options=list(predictor.models.keys()),
            key='batch_model'
        )
        
        show_details = st.checkbox("Show detailed results", value=False)
    
    if uploaded_file is not None:
        try:
            # Read data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
            
            # Predict button
            if st.button("🔮 Predict Batch", type="primary", width='stretch'):
                with st.spinner(f"Making predictions on {len(df):,} instances..."):
                    progress_bar = st.progress(0)
                    
                    # Make batch predictions
                    results = predictor.predict_batch(df, model_name=model_choice)
                    
                    progress_bar.progress(100)
                
                # Check for errors
                if results and 'error' in results[0]:
                    st.error(f"❌ {results[0]['error']}")
                    return
                
                st.success(f"✅ Completed {len(results):,} predictions!")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(results)
                
                # Summary statistics
                st.subheader("📈 Batch Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", f"{len(results_df):,}")
                with col2:
                    attacks = (results_df['prediction'] == 1).sum()
                    st.metric("Attacks Detected", f"{attacks:,}", 
                             delta=f"{attacks/len(results_df)*100:.1f}%")
                with col3:
                    normal = (results_df['prediction'] == 0).sum()
                    st.metric("Normal Traffic", f"{normal:,}",
                             delta=f"{normal/len(results_df)*100:.1f}%")
                with col4:
                    avg_conf = results_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Normal', 'Attack'],
                        values=[normal, attacks],
                        hole=0.4,
                        marker_colors=['#00CC96', '#EF553B']
                    )])
                    fig_pie.update_layout(title="Traffic Distribution", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    fig_conf = px.histogram(
                        results_df,
                        x='confidence',
                        color='label',
                        title='Confidence Distribution',
                        nbins=30,
                        color_discrete_map={'Normal': '#00CC96', 'Attack': '#EF553B'}
                    )
                    fig_conf.update_layout(height=400)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Detailed results table
                if show_details:
                    st.subheader("📋 Detailed Results")
                    st.dataframe(
                        results_df.style.format({
                            'confidence': '{:.4f}',
                            'attack_probability': '{:.4f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                
                # Download results
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Add results to original dataframe
                    output_df = df.copy()
                    output_df['Prediction'] = results_df['label']
                    output_df['Confidence'] = results_df['confidence']
                    output_df['Attack_Probability'] = results_df['attack_probability']
                    
                    csv = output_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                
                with col2:
                    # Download summary
                    summary_csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary_csv,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("💡 Upload a CSV file to make batch predictions")

def live_monitoring_tab(predictor):
    """Live monitoring interface"""
    st.subheader("🔄 Live Network Monitoring")
    
    st.markdown("""
    Monitor network traffic in real-time. Upload a CSV file to simulate live traffic.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload network traffic CSV",
            type=['csv'],
            key='live_upload'
        )
    
    with col2:
        model_choice = st.selectbox(
            "Model",
            options=['Ensemble'] + list(predictor.models.keys()),
            key='live_model'
        )
    
    with col3:
        refresh_rate = st.number_input(
            "Refresh Rate (s)",
            min_value=1,
            max_value=60,
            value=5,
            step=1
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} network flows")
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Start Monitoring", width='stretch'):
                    st.session_state.auto_refresh = True
                    st.session_state.monitor_index = 0
                    st.session_state.live_predictions = []
            
            with col2:
                if st.button("⏸️ Pause", width='stretch'):
                    st.session_state.auto_refresh = False
            
            with col3:
                if st.button("🔄 Reset", width='stretch'):
                    st.session_state.auto_refresh = False
                    st.session_state.monitor_index = 0
                    st.session_state.live_predictions = []
                    st.rerun()
            
            # Monitoring display
            if hasattr(st.session_state, 'auto_refresh') and st.session_state.auto_refresh:
                # Progress
                if not hasattr(st.session_state, 'monitor_index'):
                    st.session_state.monitor_index = 0
                if not hasattr(st.session_state, 'live_predictions'):
                    st.session_state.live_predictions = []
                
                progress = st.session_state.monitor_index / len(df)
                st.progress(progress)
                
                # Process next batch
                if st.session_state.monitor_index < len(df):
                    current_flow = df.iloc[[st.session_state.monitor_index]]
                    
                    # Make prediction
                    if model_choice == 'Ensemble':
                        result = predictor.predict_ensemble(current_flow, voting='soft')
                    else:
                        result = predictor.predict_single(current_flow, model_name=model_choice)
                    
                    if 'error' not in result:
                        st.session_state.live_predictions.append(result)
                        st.session_state.monitor_index += 1
                    
                    # Display live stats
                    display_live_stats(st.session_state.live_predictions)
                    
                    # Auto-refresh
                    time.sleep(refresh_rate)
                    st.rerun()
                else:
                    st.success("✅ Monitoring complete!")
                    st.session_state.auto_refresh = False
            
            # Display current stats if available
            elif hasattr(st.session_state, 'live_predictions') and st.session_state.live_predictions:
                display_live_stats(st.session_state.live_predictions)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("💡 Upload a CSV file to start live monitoring")

def statistics_tab(predictor):
    """Statistics and history display"""
    st.subheader("📈 Prediction Statistics")
    
    # Get statistics
    stats = predictor.get_statistics()
    
    if 'message' in stats:
        st.info(stats['message'])
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{stats['total_predictions']:,}")
    with col2:
        st.metric("Attacks Detected", f"{stats['attacks_detected']:,}",
                 delta=f"{stats['attack_rate']*100:.1f}%")
    with col3:
        st.metric("Normal Traffic", f"{stats['normal_traffic']:,}")
    with col4:
        st.metric("Avg Confidence", f"{stats['average_confidence']:.3f}")
    
    st.markdown("---")
    
    # History visualization
    if predictor.prediction_history:
        history_df = pd.DataFrame(predictor.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Time series
        st.subheader("🕐 Prediction Timeline")
        
        fig_timeline = go.Figure()
        
        for label, color in [('Normal', '#00CC96'), ('Attack', '#EF553B')]:
            mask = history_df['label'] == label
            fig_timeline.add_trace(go.Scatter(
                x=history_df[mask]['timestamp'],
                y=history_df[mask]['attack_probability'],
                mode='markers',
                name=label,
                marker=dict(color=color, size=8)
            ))
        
        fig_timeline.update_layout(
            title="Attack Probability Over Time",
            xaxis_title="Time",
            yaxis_title="Attack Probability",
            height=400
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        
        recent_df = history_df[['timestamp', 'label', 'confidence', 'attack_probability']].tail(20)
        st.dataframe(
            recent_df.style.format({
                'confidence': '{:.4f}',
                'attack_probability': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export History", width='stretch'):
                ensure_directory_exists("data/predictions/")
                if predictor.export_history():
                    st.success("✅ History exported successfully!")
        
        with col2:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col3:
            if st.button("🗑️ Clear History", width='stretch'):
                predictor.clear_history()
                st.success("✅ History cleared!")
                st.rerun()

def display_prediction_result(result, original_data=None):
    """Display detailed prediction result"""
    
    # Main result card
    if result['label'] == 'Attack':
        st.error("🚨 **ATTACK DETECTED!**")
    else:
        st.success("✅ **Normal Traffic**")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", result['label'])
    with col2:
        st.metric("Confidence", f"{result['confidence']:.2%}")
    with col3:
        st.metric("Attack Probability", f"{result['attack_probability']:.2%}")
    with col4:
        if 'model_used' in result:
            st.metric("Model", result['model_used'])
    
    # Confidence gauge
    st.subheader("📊 Confidence Level")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['confidence'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Ensemble details if available
    if 'individual_models' in result:
        st.subheader("🎯 Individual Model Results")
        
        models_df = pd.DataFrame.from_dict(result['individual_models'], orient='index')
        models_df.index.name = 'Model'
        models_df.reset_index(inplace=True)
        models_df['prediction_label'] = models_df['prediction'].apply(lambda x: 'Attack' if x == 1 else 'Normal')
        
        # Display table
        st.dataframe(
            models_df.style.format({
                'attack_probability': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Visualization
        fig_models = go.Figure()
        
        fig_models.add_trace(go.Bar(
            x=models_df['Model'],
            y=models_df['attack_probability'],
            text=models_df['attack_probability'].apply(lambda x: f"{x:.3f}"),
            textposition='auto',
            marker_color=models_df['prediction'].apply(lambda x: '#EF553B' if x == 1 else '#00CC96')
        ))
        
        fig_models.update_layout(
            title="Attack Probability by Model",
            xaxis_title="Model",
            yaxis_title="Attack Probability",
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig_models, use_container_width=True)
    
    # Original data preview
    if original_data is not None:
        with st.expander("📋 View Original Data"):
            st.dataframe(original_data, use_container_width=True)
    
    # Timestamp
    st.caption(f"Prediction made at: {result['timestamp']}")

def display_live_stats(predictions):
    """Display live monitoring statistics"""
    
    if not predictions:
        return
    
    st.subheader("📊 Live Statistics")
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(pred_df)
    attacks = (pred_df['prediction'] == 1).sum()
    normal = (pred_df['prediction'] == 0).sum()
    avg_conf = pred_df['confidence'].mean()
    
    with col1:
        st.metric("Flows Analyzed", f"{total:,}")
    with col2:
        st.metric("Attacks", f"{attacks:,}", delta=f"{attacks/total*100:.1f}%")
    with col3:
        st.metric("Normal", f"{normal:,}", delta=f"{normal/total*100:.1f}%")
    with col4:
        st.metric("Avg Confidence", f"{avg_conf:.3f}")
    
    # Real-time chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Latest predictions
        st.subheader("🔄 Latest Detections")
        
        latest = pred_df.tail(10)[['label', 'confidence', 'attack_probability']]
        st.dataframe(
            latest.style.format({
                'confidence': '{:.3f}',
                'attack_probability': '{:.3f}'
            }),
            use_container_width=True
        )
    
    with col2:
        # Attack rate over time
        st.subheader("📈 Attack Rate Trend")
        
        # Calculate rolling attack rate
        window_size = min(10, len(pred_df))
        pred_df['rolling_attack_rate'] = pred_df['prediction'].rolling(window=window_size, min_periods=1).mean()
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            y=pred_df['rolling_attack_rate'] * 100,
            mode='lines+markers',
            name='Attack Rate',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig_trend.update_layout(
            xaxis_title="Sample",
            yaxis_title="Attack Rate (%)",
            yaxis_range=[0, 100],
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Alert for high attack rate
    if attacks / total > 0.5:
        st.warning("⚠️ **High Attack Rate Detected!** More than 50% of traffic classified as attacks.")
    
    # Latest prediction highlight
    latest_pred = predictions[-1]
    if latest_pred['label'] == 'Attack':
        st.error(f"🚨 Latest: **ATTACK** detected (Confidence: {latest_pred['confidence']:.2%})")
    else:
        st.success(f"✅ Latest: **Normal** traffic (Confidence: {latest_pred['confidence']:.2%})")

if __name__ == "__main__":
    main()