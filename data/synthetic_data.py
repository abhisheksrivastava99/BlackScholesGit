"""
data/synthetic_data.py
Generate synthetic Black-Scholes option pricing data for training neural networks.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from typing import Tuple, Optional


class BlackScholesDataGenerator:
    """
    Generate synthetic option pricing data using the Black-Scholes formula.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    @staticmethod
    def black_scholes_call(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                          r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                         r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def generate_parameters(self, n_samples: int,
                          S_range: Tuple[float, float] = (50.0, 150.0),
                          K_range: Tuple[float, float] = (50.0, 150.0),
                          T_range: Tuple[float, float] = (0.1, 2.0),
                          r_range: Tuple[float, float] = (0.01, 0.1),
                          sigma_range: Tuple[float, float] = (0.1, 0.5)) -> pd.DataFrame:
        """
        Generate random parameter combinations.
        
        Args:
            n_samples: Number of samples to generate
            S_range: Range for stock price (min, max)
            K_range: Range for strike price (min, max)
            T_range: Range for time to expiration (min, max)
            r_range: Range for risk-free rate (min, max)
            sigma_range: Range for volatility (min, max)
            
        Returns:
            DataFrame with generated parameters
        """
        S = np.random.uniform(S_range[0], S_range[1], n_samples)
        K = np.random.uniform(K_range[0], K_range[1], n_samples)
        T = np.random.uniform(T_range[0], T_range[1], n_samples)
        r = np.random.uniform(r_range[0], r_range[1], n_samples)
        sigma = np.random.uniform(sigma_range[0], sigma_range[1], n_samples)
        
        df = pd.DataFrame({
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma
        })
        
        return df
    
    
    def generate_dataset(self, n_samples: int = 100000,
                     add_noise: bool = False,
                     noise_level: float = 0.01,
                     **param_ranges) -> pd.DataFrame:
        """
        Generate complete dataset with parameters and option prices
        for BOTH calls and puts, using option_type as an extra feature.

        Returns:
            DataFrame with parameters and option prices
        """
        # Generate random parameters
        df = self.generate_parameters(n_samples, **param_ranges)

        # Randomly assign each row to be a call or put
        option_types = np.random.choice(["call", "put"], n_samples)
        option_type_feature = (option_types == "put").astype(int)  # 0=call, 1=put

        # Compute prices for each
        prices = []
        for i in range(n_samples):
            if option_types[i] == "call":
                price = self.black_scholes_call(
                    df["S"].values[i],
                    df["K"].values[i],
                    df["T"].values[i],
                    df["r"].values[i],
                    df["sigma"].values[i]
                )
            else:
                price = self.black_scholes_put(
                    df["S"].values[i],
                    df["K"].values[i],
                    df["T"].values[i],
                    df["r"].values[i],
                    df["sigma"].values[i]
                )
            if add_noise:
                price += np.random.normal(0, noise_level * price)
            prices.append(max(price, 0))  # Ensure non-negative

        # Add to DataFrame
        df["option_type"] = option_type_feature
        df["price"] = prices
        return df
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                        shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Fraction of data for testing
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if shuffle:
            df = df.sample(frac=1.0).reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df[:split_idx].reset_index(drop=True)
        test_df = df[split_idx:].reset_index(drop=True)
        
        return train_df, test_df
    
    def to_torch_tensors(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert DataFrame to PyTorch tensors. Input features include option_type!
        """
        features = df[["S", "K", "T", "r", "sigma", "option_type"]].values
        target = df["price"].values
        features_tensor = torch.FloatTensor(features)
        target_tensor = torch.FloatTensor(target).reshape(-1, 1)
        return features_tensor, target_tensor



# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = BlackScholesDataGenerator(seed=42)
    
    # Generate dataset
    print("Generating synthetic option pricing data...")
    data = generator.generate_dataset(
        n_samples=100000,
        add_noise=False
    )
    
    print(f"\nGenerated {len(data)} samples")
    print("\nFirst few rows:")
    print(data.head())
    
    print("\nDataset statistics:")
    print(data.describe())
    
    # Split into train/test
    train_data, test_data = generator.train_test_split(data, test_size=0.2)
    print(f"\nTrain set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Convert to PyTorch tensors
    X_train, y_train = generator.to_torch_tensors(train_data)
    X_test, y_test = generator.to_torch_tensors(test_data)
    
    print(f"\nPyTorch tensor shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Save to CSV (optional)generate_parameter
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    print("\nData saved to train_data.csv and test_data.csv")
