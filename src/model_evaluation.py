# src/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, roc_auc_score, auc
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    def __init__(self, trainer):
        """
        Initialize evaluator with trained models
        
        Args:
            trainer: ModelTrainer instance with trained models
        """
        self.trainer = trainer
        self.results = trainer.training_results
    
    def plot_confusion_matrix(self, model_name, normalize=False, use_plotly=True):
        """
        Plot confusion matrix for a specific model
        
        Args:
            model_name: Name of the model
            normalize: Whether to normalize the matrix
            use_plotly: Use plotly (True) or matplotlib (False)
        """
        if model_name not in self.results:
            print(f"❌ Model '{model_name}' not found!")
            return None
        
        cm = self.results[model_name]['test_confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate percentages and counts
        tn, fp, fn, tp = cm.ravel() if not normalize else (cm * np.sum(self.results[model_name]['test_confusion_matrix'])).ravel()
        total = tn + fp + fn + tp
        
        if use_plotly:
            # Plotly version
            labels = ['Normal', 'Attack']
            
            # Create text annotations
            text = [[f"TN<br>{tn:.0f}<br>({tn/total*100:.2f}%)", 
                    f"FP<br>{fp:.0f}<br>({fp/total*100:.2f}%)"],
                   [f"FN<br>{fn:.0f}<br>({fn/total*100:.2f}%)", 
                    f"TP<br>{tp:.0f}<br>({tp/total*100:.2f}%)"]]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=f'Confusion Matrix - {model_name}',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                width=500,
                height=500
            )
            
            return fig
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2%' if normalize else 'd',
                       cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xticklabels(['Normal', 'Attack'])
            ax.set_yticklabels(['Normal', 'Attack'])
            
            return fig
    
    def plot_all_confusion_matrices(self):
        """Plot confusion matrices for all models in a grid"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.results.keys()),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        labels = ['Normal', 'Attack']
        
        for idx, (model_name, pos) in enumerate(zip(self.results.keys(), positions)):
            cm = self.results[model_name]['test_confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            text = [[f"TN: {tn}<br>{tn/total*100:.1f}%", 
                    f"FP: {fp}<br>{fp/total*100:.1f}%"],
                   [f"FN: {fn}<br>{fn/total*100:.1f}%", 
                    f"TP: {tp}<br>{tp/total*100:.1f}%"]]
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    text=text,
                    texttemplate="%{text}",
                    colorscale='Blues',
                    showscale=False
                ),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(
            title_text='Confusion Matrices - All Models',
            height=800,
            width=900
        )
        
        return fig
    
    def plot_roc_curves(self, show_all=True):
        """
        Plot ROC curves for all models
        
        Args:
            show_all: Show all models on one plot (True) or separate (False)
        """
        fig = go.Figure()
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if 'roc_curve' in results and results['roc_curve'] is not None:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                auc_score = results.get('test_roc_auc', auc(fpr, tpr))
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{model_name} (AUC = {auc_score:.4f})",
                    line=dict(color=colors[idx], width=2)
                ))
        
        fig.update_layout(
            title='ROC Curves - All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_metrics_comparison(self, metric='test_accuracy'):
        """
        Plot bar chart comparing all models on a specific metric
        
        Args:
            metric: Metric to compare ('test_accuracy', 'test_precision', etc.)
        """
        models = list(self.results.keys())
        values = [self.results[m][metric] for m in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=values,
                text=[f'{v:.4f}' for v in values],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
        ])
        
        metric_name = metric.replace('test_', '').replace('_', ' ').title()
        
        fig.update_layout(
            title=f'{metric_name} Comparison Across Models',
            xaxis_title='Model',
            yaxis_title=metric_name,
            yaxis_range=[0, 1.05],
            height=500
        )
        
        return fig
    
    def plot_all_metrics_comparison(self):
        """Plot comparison of all metrics across all models"""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names
        )
        
        models = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, (metric, name, pos) in enumerate(zip(metrics, metric_names, positions)):
            values = [self.results[m][metric] for m in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    text=[f'{v:.4f}' for v in values],
                    textposition='auto',
                    marker_color=colors,
                    showlegend=False
                ),
                row=pos[0], col=pos[1]
            )
            
            fig.update_yaxes(range=[0, 1.05], row=pos[0], col=pos[1])
        
        fig.update_layout(
            title_text='Performance Metrics Comparison - All Models',
            height=800,
            width=1000
        )
        
        return fig
    
    def get_classification_report(self, model_name):
        """Get detailed classification report for a model"""
        if model_name not in self.results:
            return None
        
        y_true = self.trainer.y_test
        y_pred = self.results[model_name]['y_test_pred']
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Attack'],
            output_dict=True
        )
        
        return pd.DataFrame(report).transpose()
    
    def generate_evaluation_summary(self):
        """Generate comprehensive evaluation summary"""
        summary = {
            'Model Performance Summary': self.trainer.get_comparison_dataframe(),
            'Best Model': None,
            'Detailed Reports': {}
        }
        
        # Find best model
        best_accuracy = max(r['test_accuracy'] for r in self.results.values())
        best_model = [name for name, r in self.results.items() 
                     if r['test_accuracy'] == best_accuracy][0]
        
        summary['Best Model'] = {
            'Name': best_model,
            'Accuracy': self.results[best_model]['test_accuracy'],
            'Precision': self.results[best_model]['test_precision'],
            'Recall': self.results[best_model]['test_recall'],
            'F1-Score': self.results[best_model]['test_f1']
        }
        
        # Detailed reports for each model
        for model_name in self.results.keys():
            summary['Detailed Reports'][model_name] = self.get_classification_report(model_name)
        
        return summary