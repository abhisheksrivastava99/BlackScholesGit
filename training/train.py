"""
training/train.py
Training script for Black-Scholes neural network approximator.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import OptionPricingNN, InputNormalizer
from models.black_scholes import BlackScholesModel
from data.synthetic_data import BlackScholesDataGenerator

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improvement
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered!')
        else:
            if self.verbose:
                print(f'Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f})')
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """
    Trainer class for Black-Scholes neural network.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization parameter
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function (Mean Squared Error for regression)
        self.criterion = nn.MSELoss()
        
        # Optimizer (Adam with weight decay)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move data to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features)
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, average MAPE)
        """
        self.model.eval()
        total_loss = 0.0
        total_mape = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                
                # Calculate loss
                loss = self.criterion(predictions, targets)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100
                
                total_loss += loss.item()
                total_mape += mape.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mape = total_mape / num_batches
        
        return avg_loss, avg_mape
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            early_stopping_patience: int = 15,
            save_path: Optional[str] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-6,
            verbose=verbose
        )
        
        if verbose:
            print("="*80)
            print("Starting Training")
            print("="*80)
            print(f"Device: {self.device}")
            print(f"Model parameters: {self.model.get_num_parameters():,}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            print("="*80)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mape = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAPE: {val_mape:.2f}% | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                # Restore best model
                self.model.load_state_dict(early_stopping.best_model_state)
                break
        
        # Save best model
        if save_path is not None:
            self.save_model(save_path)
            if verbose:
                print(f"\nBest model saved to: {save_path}")
        
        if verbose:
            print("="*80)
            print("Training Complete!")
            print(f"Best Validation Loss: {early_stopping.best_loss:.6f}")
            print("="*80)
        
        return self.history
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        save_path = 'saved_models/best_model.pth'
        save_dir = os.path.dirname(save_path)  # save_path = '../BlackSholes/saved_models/your_model.pth'
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_model(path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        self.model = OptionPricingNN.load_model(path, self.device)


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Learning rate
    axes[0, 1].plot(history['learning_rate'], color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Epoch time
    axes[1, 0].plot(history['epoch_time'], color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, color='red', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].set_title('Overfitting Monitor')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


# Main training script
if __name__ == "__main__":
    print("\n" + "="*80)
    print("BLACK-SCHOLES NEURAL NETWORK TRAINER")
    print("="*80 + "\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs('saved_models/plots', exist_ok=True)
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES = 100000
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    print(f"Device: {DEVICE}")
    print(f"Samples: {N_SAMPLES:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = BlackScholesDataGenerator(seed=42)
    data = generator.generate_dataset(
        n_samples=N_SAMPLES,
        add_noise=False
    )
    
    # Split data
    train_data, test_data = generator.train_test_split(data, test_size=0.2)
    
    # Convert to tensors
    X_train, y_train = generator.to_torch_tensors(train_data)
    X_test, y_test = generator.to_torch_tensors(test_data)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}\n")
    
    # Normalize features
    print("Normalizing features...")
    normalizer = InputNormalizer()
    normalizer.fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_norm, y_train)
    test_dataset = TensorDataset(X_test_norm, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = OptionPricingNN(
        input_dim=6,
        hidden_dims=[128, 128, 64],
        output_dim=1,
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Train model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=EPOCHS,
        early_stopping_patience=15,
        save_path='saved_models/best_model.pth',
        verbose=True
    )
    
    # Plot training history
    plot_training_history(history, save_path='saved_models/training_history.png')
    
    # Save normalizer
    os.makedirs('saved_models', exist_ok=True)
    normalizer.save('saved_models/normalizer.pth')
    print("\nNormalizer saved to: saved_models/normalizer.pth")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
