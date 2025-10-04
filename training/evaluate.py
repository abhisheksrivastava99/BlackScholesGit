"""
training/evaluate.py
Comprehensive evaluation script for Black-Scholes neural network approximator.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import OptionPricingNN, InputNormalizer
from models.black_scholes import BlackScholesModel
from data.synthetic_data import BlackScholesDataGenerator


class ModelEvaluator:
    """
    Comprehensive evaluation for option pricing neural networks.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained neural network model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.bs_model = BlackScholesModel(device=device)
    
    def calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        # Mean Squared Error (MSE)
        mse = mean_squared_error(y_true_np, y_pred_np)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true_np > 0.01  # Only calculate MAPE for option prices > $0.01
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100
        else:
            mape = np.inf  # Undefined if no valid points   
        
        # R-squared (R²)
        r2 = r2_score(y_true_np, y_pred_np)
        
        # Max Absolute Error
        max_error = np.max(np.abs(y_true_np - y_pred_np))
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true_np - y_pred_np))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Max_Error': max_error,
            'Median_AE': median_ae
        }
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            Tuple of (predictions, targets, metrics)
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(targets, predictions)
        
        return predictions, targets, metrics
    
    def compare_with_black_scholes(self, 
                                   features: torch.Tensor,
                                   nn_predictions: torch.Tensor,
                                   option_type: str = 'call') -> Dict[str, float]:
        """
        Compare neural network predictions with analytical Black-Scholes.
        
        Args:
            features: Input features (S, K, T, r, sigma)
            nn_predictions: Neural network predictions
            option_type: 'call' or 'put'
            
        Returns:
            Comparison metrics
        """
        # Calculate Black-Scholes prices
        S = features[:, 0]
        K = features[:, 1]
        T = features[:, 2]
        r = features[:, 3]
        sigma = features[:, 4]
        
        bs_predictions = self.bs_model.batch_price(S, K, T, r, sigma, option_type)
        bs_predictions = bs_predictions.reshape(-1, 1)
        
        # Calculate comparison metrics
        comparison_metrics = self.calculate_metrics(bs_predictions, nn_predictions)
        
        return comparison_metrics, bs_predictions
    
    def plot_predictions(self, 
                        y_true: torch.Tensor, 
                        y_pred: torch.Tensor,
                        save_path: Optional[str] = None):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot
        """
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        axes[0].scatter(y_true_np, y_pred_np, alpha=0.5, s=10)
        axes[0].plot([y_true_np.min(), y_true_np.max()], 
                     [y_true_np.min(), y_true_np.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('True Option Price', fontsize=12)
        axes[0].set_ylabel('Predicted Option Price', fontsize=12)
        axes[0].set_title('Predicted vs Actual Option Prices', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true_np - y_pred_np
        axes[1].scatter(y_pred_np, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Option Price', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to: {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self,
                               y_true: torch.Tensor,
                               y_pred: torch.Tensor,
                               save_path: Optional[str] = None):
        """
        Plot error distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot
        """
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        errors = y_true_np - y_pred_np
        percentage_errors = (errors / (y_true_np + 1e-8)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute error histogram
        axes[0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(np.abs(errors)), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(np.abs(errors)):.4f}')
        axes[0].axvline(np.median(np.abs(errors)), color='g', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(np.abs(errors)):.4f}')
        axes[0].set_xlabel('Absolute Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Percentage error histogram
        axes[1].hist(np.abs(percentage_errors), bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(np.abs(percentage_errors)), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(np.abs(percentage_errors)):.2f}%')
        axes[1].axvline(np.median(np.abs(percentage_errors)), color='g', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(np.abs(percentage_errors)):.2f}%')
        axes[1].set_xlabel('Absolute Percentage Error (%)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_comparison_with_bs(self,
                               features: torch.Tensor,
                               nn_predictions: torch.Tensor,
                               bs_predictions: torch.Tensor,
                               save_path: Optional[str] = None):
        """
        Plot comparison between NN and Black-Scholes predictions.
        
        Args:
            features: Input features
            nn_predictions: Neural network predictions
            bs_predictions: Black-Scholes predictions
            save_path: Path to save plot
        """
        nn_pred_np = nn_predictions.cpu().numpy().flatten()
        bs_pred_np = bs_predictions.cpu().numpy().flatten()
        
        # Calculate errors relative to Black-Scholes
        errors = nn_pred_np - bs_pred_np
        percentage_errors = (errors / (bs_pred_np + 1e-8)) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # NN vs BS scatter
        axes[0, 0].scatter(bs_pred_np, nn_pred_np, alpha=0.5, s=10)
        axes[0, 0].plot([bs_pred_np.min(), bs_pred_np.max()],
                       [bs_pred_np.min(), bs_pred_np.max()],
                       'r--', lw=2, label='Perfect Match')
        axes[0, 0].set_xlabel('Black-Scholes Price', fontsize=12)
        axes[0, 0].set_ylabel('Neural Network Price', fontsize=12)
        axes[0, 0].set_title('NN vs Black-Scholes Predictions', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error vs BS price
        axes[0, 1].scatter(bs_pred_np, errors, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Black-Scholes Price', fontsize=12)
        axes[0, 1].set_ylabel('Error (NN - BS)', fontsize=12)
        axes[0, 1].set_title('Error vs Black-Scholes Price', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Absolute error distribution
        axes[1, 0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(np.abs(errors)), color='r', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(np.abs(errors)):.6f}')
        axes[1, 0].set_xlabel('Absolute Error', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Absolute Error Distribution (NN vs BS)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Percentage error distribution
        axes[1, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(percentage_errors), color='r', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(percentage_errors):.4f}%')
        axes[1, 1].set_xlabel('Percentage Error (%)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BS comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, 
                                  metrics: Dict[str, float],
                                  comparison_metrics: Dict[str, float],
                                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Model evaluation metrics
            comparison_metrics: Comparison with Black-Scholes
            save_path: Path to save report
            
        Returns:
            DataFrame with evaluation results
        """
        report_data = {
            'Metric': [],
            'Value': [],
            'Description': []
        }
        
        metric_descriptions = {
            'MSE': 'Mean Squared Error - Average squared difference',
            'RMSE': 'Root Mean Squared Error - Standard deviation of residuals',
            'MAE': 'Mean Absolute Error - Average absolute difference',
            'MAPE': 'Mean Absolute Percentage Error - Average percentage error',
            'R2': 'R-squared - Proportion of variance explained',
            'Max_Error': 'Maximum Absolute Error - Worst case error',
            'Median_AE': 'Median Absolute Error - 50th percentile of errors'
        }
        
        print("\n" + "="*80)
        print("NEURAL NETWORK EVALUATION REPORT")
        print("="*80)
        
        print("\n1. Model Performance Metrics:")
        print("-"*80)
        for metric, value in metrics.items():
            report_data['Metric'].append(metric)
            report_data['Value'].append(value)
            report_data['Description'].append(metric_descriptions.get(metric, ''))
            
            if metric == 'MAPE':
                print(f"{metric:15s}: {value:12.4f}%")
            elif metric == 'R2':
                print(f"{metric:15s}: {value:12.6f}")
            else:
                print(f"{metric:15s}: {value:12.6f}")
        
        print("\n2. Comparison with Black-Scholes:")
        print("-"*80)
        for metric, value in comparison_metrics.items():
            if metric == 'MAPE':
                print(f"{metric:15s}: {value:12.4f}% (vs analytical BS)")
            elif metric == 'R2':
                print(f"{metric:15s}: {value:12.6f} (vs analytical BS)")
            else:
                print(f"{metric:15s}: {value:12.6f} (vs analytical BS)")
        
        print("="*80)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"\nEvaluation report saved to: {save_path}")
        
        return report_df


# Main evaluation script
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BLACK-SCHOLES NEURAL NETWORK EVALUATION")
    print("="*80 + "\n")
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'saved_models/best_model.pth'
    NORMALIZER_PATH = 'saved_models/normalizer.pth'
    N_TEST_SAMPLES = 20000
    
    print(f"Device: {DEVICE}")
    print(f"Test samples: {N_TEST_SAMPLES:,}\n")
    
    # Generate test data
    print("Generating test data...")
    generator = BlackScholesDataGenerator(seed=123)  # Different seed for test
    test_data = generator.generate_dataset(
        n_samples=N_TEST_SAMPLES,
        add_noise=False
    )
    
    X_test, y_test = generator.to_torch_tensors(test_data)
    
    # Load normalizer and normalize features
    print("Loading normalizer...")
    normalizer = InputNormalizer()
    normalizer.load(NORMALIZER_PATH)
    X_test_norm = normalizer.transform(X_test)
    
    # Create data loader
    test_dataset = TensorDataset(X_test_norm, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Load model
    print("Loading trained model...")
    model = OptionPricingNN.load_model(MODEL_PATH, device=DEVICE)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device=DEVICE)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, targets, metrics = evaluator.evaluate(test_loader)
    
    # Compare with Black-Scholes
    print("Comparing with Black-Scholes analytical solution...")
    #comparison_metrics, bs_predictions = evaluator.compare_with_black_scholes(
    #    X_test, predictions, option_type='call')
    option_types = test_data['option_type'].cpu().numpy() if hasattr(test_data['option_type'], 'cpu') else test_data['option_type'].values

    bs_preds = []
    for i in range(len(X_test)):
        opttype = 'put' if option_types[i] == 1 else 'call'
        _, bs_pred = evaluator.compare_with_black_scholes(
            X_test[i:i+1], predictions[i:i+1], option_type=opttype
        )
        bs_preds.append(bs_pred)

    bs_predictions = torch.cat(bs_preds, dim=0)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_bs_all = bs_predictions.cpu().numpy().flatten()
    y_nn_all = predictions.cpu().numpy().flatten()
    r2 = r2_score(y_bs_all, y_nn_all)
    mae = mean_absolute_error(y_bs_all, y_nn_all)
    mse = mean_squared_error(y_bs_all, y_nn_all)
    comparison_metrics = {"R2": r2, "MAE": mae, "MSE": mse}
    print(f"\nSummary Metrics vs Black-Scholes (All Samples):")
    print(f"R²: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")
    # Generate report
    report = evaluator.generate_evaluation_report(
        metrics,
        comparison_metrics,
        
        save_path='saved_models/evaluation_report.csv'
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    os.makedirs('saved_models/plots', exist_ok=True)
    
    evaluator.plot_predictions(
        targets, predictions,
        save_path='saved_models/plots/predictions.png'
    )
    
    evaluator.plot_error_distribution(
        targets, predictions,
        os.makedirs('saved_models/plots/', exist_ok=True),
        save_path='saved_models/plots/error_distribution.png'
    )
    
    evaluator.plot_comparison_with_bs(
        X_test, predictions, bs_predictions,
        os.makedirs('saved_models/plots/bs_comparison.png', exist_ok=True),
        save_path='.saved_models/plots/bs_comparison.png'
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
