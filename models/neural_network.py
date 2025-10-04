"""
models/neural_network.py
PyTorch neural network architecture for approximating Black-Scholes option pricing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class OptionPricingNN(nn.Module):
    """
    Feedforward Neural Network for Black-Scholes option price approximation.
    
    Architecture:
    - Input layer: 5 features (S, K, T, r, sigma)
    - Hidden layers: Configurable depth and width
    - Output layer: 1 value (option price)
    - Activation: ELU (for smooth gradients, important for Greeks calculation)
    """
    
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dims: List[int] = [128, 128, 64],
                 output_dim: int = 1,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of input features (default: 5 for S, K, T, r, sigma)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output values (default: 1 for option price)
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(OptionPricingNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function (ELU for smooth higher-order derivatives)
            layers.append(nn.ELU())
            
            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation - regression task)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Features: [S, K, T, r, sigma]
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
            Predicted option price
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted option prices
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path: str):
        """
        Save model weights and architecture.
        
        Args:
            path: File path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """
        Load model from file.
        
        Args:
            path: File path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            output_dim=checkpoint['output_dim'],
            dropout_rate=checkpoint['dropout_rate'],
            use_batch_norm=checkpoint['use_batch_norm']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class DeepOptionPricingNN(nn.Module):
    """
    Deeper neural network architecture for complex option pricing tasks.
    
    This architecture uses:
    - Residual connections for better gradient flow
    - Layer normalization instead of batch normalization
    - Larger capacity for learning complex patterns
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 hidden_dim: int = 256,
                 num_layers: int = 5,
                 output_dim: int = 1,
                 dropout_rate: float = 0.1):
        """
        Initialize deep neural network with residual connections.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of residual blocks
            output_dim: Number of output values
            dropout_rate: Dropout probability
        """
        super(DeepOptionPricingNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Input projection
        x = self.input_layer(x)
        x = F.elu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output projection
        x = self.output_layer(x)
        
        return x
    
    def save_model(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            output_dim=checkpoint['output_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        # First sublayer
        x = self.layer_norm1(x)
        x = self.linear1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second sublayer
        x = self.layer_norm2(x)
        x = self.linear2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class InputNormalizer:
    """
    Normalize input features for better neural network training.
    
    Stores mean and std for each feature to normalize/denormalize data.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X: torch.Tensor):
        """
        Compute mean and std from training data.
        
        Args:
            X: Training data tensor
        """
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        self.fitted = True
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using stored statistics.
        
        Args:
            X: Input tensor
            
        Returns:
            Normalized tensor
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            X: Normalized tensor
            
        Returns:
            Original scale tensor
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        return X * self.std + self.mean
    
    def save(self, path: str):
        """Save normalizer statistics."""
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }, path)
    
    def load(self, path: str):
        """Load normalizer statistics."""
        checkpoint = torch.load(path)
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.fitted = checkpoint['fitted']


# Example usage and testing
if __name__ == "__main__":
    print("Option Pricing Neural Network Architecture\n" + "="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example 1: Standard feedforward network
    print("\n1. Standard Feedforward Network:")
    print("-" * 60)
    model = OptionPricingNN(
        input_dim=6,
        hidden_dims=[128, 128, 64],
        output_dim=1,
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    print(f"Model Architecture:\n{model}")
    print(f"\nTotal Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 5)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Example 2: Deep residual network
    print("\n" + "="*60)
    print("\n2. Deep Residual Network:")
    print("-" * 60)
    deep_model = DeepOptionPricingNN(
        input_dim=6,
        hidden_dim=256,
        num_layers=5,
        output_dim=1,
        dropout_rate=0.1
    )
    
    print(f"Deep Model Architecture:\n{deep_model}")
    
    output_deep = deep_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output_deep.shape}")
    
    # Example 3: Input normalization
    print("\n" + "="*60)
    print("\n3. Input Normalization:")
    print("-" * 60)
    
    # Create synthetic data
    X_train = torch.randn(1000, 5) * 50 + 100  # Mean=100, varied scale
    
    print(f"Original data statistics:")
    print(f"Mean: {X_train.mean(dim=0)}")
    print(f"Std:  {X_train.std(dim=0)}")
    
    # Fit normalizer
    normalizer = InputNormalizer()
    normalizer.fit(X_train)
    
    # Transform data
    X_train_normalized = normalizer.transform(X_train)
    
    print(f"\nNormalized data statistics:")
    print(f"Mean: {X_train_normalized.mean(dim=0)}")
    print(f"Std:  {X_train_normalized.std(dim=0)}")
    
    # Test inverse transform
    X_train_reconstructed = normalizer.inverse_transform(X_train_normalized)
    reconstruction_error = (X_train - X_train_reconstructed).abs().max()
    print(f"\nMax reconstruction error: {reconstruction_error:.2e}")
    
    # Example 4: Save and load model
    print("\n" + "="*60)
    print("\n4. Model Persistence:")
    print("-" * 60)
    
    # Save model
    model_path = "test_model.pth"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Load model
    loaded_model = OptionPricingNN.load_model(model_path)
    print(f"Model loaded successfully!")
    
    # Verify loaded model
    output_original = model(dummy_input)
    output_loaded = loaded_model(dummy_input)
    load_error = (output_original - output_loaded).abs().max()
    print(f"Max difference between original and loaded: {load_error:.2e}")
    
    print("\n" + "="*60)
    print("Testing complete!")
