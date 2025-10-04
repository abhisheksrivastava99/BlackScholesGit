"""
models/black_scholes.py
PyTorch implementation of Black-Scholes option pricing with automatic differentiation for Greeks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union


class BlackScholesModel:
    """
    Black-Scholes option pricing model with automatic differentiation for Greeks.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Black-Scholes model.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        # Normal distribution for CDF and PDF
        self.normal = torch.distributions.Normal(0.0, 1.0)
    
    def _cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal cumulative distribution function."""
        return self.normal.cdf(x)
    
    def _pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal probability density function."""
        return torch.exp(self.normal.log_prob(x))
    
    def _compute_d1_d2(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                       r: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute d1 and d2 terms for Black-Scholes formula.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)
        return d1, d2
    
    def call_price(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                   r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1, d2 = self._compute_d1_d2(S, K, T, r, sigma)
        call = S * self._cdf(d1) - K * torch.exp(-r * T) * self._cdf(d2)
        return call
    
    def put_price(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                  r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1, d2 = self._compute_d1_d2(S, K, T, r, sigma)
        put = K * torch.exp(-r * T) * self._cdf(-d2) - S * self._cdf(-d1)
        return put
    
    def option_price(self, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor,
                    r: torch.Tensor, sigma: torch.Tensor, 
                    option_type: str = 'call') -> torch.Tensor:
        """
        Calculate option price for either call or put.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if option_type.lower() == 'call':
            return self.call_price(S, K, T, r, sigma)
        elif option_type.lower() == 'put':
            return self.put_price(S, K, T, r, sigma)
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'.")
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks using automatic differentiation.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing option price and Greeks:
            - price: Option price
            - delta: Rate of change of option price with respect to stock price
            - gamma: Rate of change of delta with respect to stock price
            - theta: Rate of change of option price with respect to time
            - vega: Rate of change of option price with respect to volatility
            - rho: Rate of change of option price with respect to risk-free rate
        """
        # Convert to tensors with gradient tracking
        S_t = torch.tensor(S, requires_grad=True, dtype=torch.float32)
        K_t = torch.tensor(K, dtype=torch.float32)
        T_t = torch.tensor(T, requires_grad=True, dtype=torch.float32)
        r_t = torch.tensor(r, requires_grad=True, dtype=torch.float32)
        sigma_t = torch.tensor(sigma, requires_grad=True, dtype=torch.float32)
        
        # Calculate option price
        price = self.option_price(S_t, K_t, T_t, r_t, sigma_t, option_type)
        
        # First-order Greeks using autograd
        price.backward()
        
        delta = S_t.grad.item() if S_t.grad is not None else 0.0
        theta = T_t.grad.item() if T_t.grad is not None else 0.0
        vega = sigma_t.grad.item() if sigma_t.grad is not None else 0.0
        rho = r_t.grad.item() if r_t.grad is not None else 0.0
        
        # Gamma (second-order derivative) requires new computation graph
        S_t2 = torch.tensor(S, requires_grad=True, dtype=torch.float32)
        K_t2 = torch.tensor(K, dtype=torch.float32)
        T_t2 = torch.tensor(T, dtype=torch.float32)
        r_t2 = torch.tensor(r, dtype=torch.float32)
        sigma_t2 = torch.tensor(sigma, dtype=torch.float32)
        
        price2 = self.option_price(S_t2, K_t2, T_t2, r_t2, sigma_t2, option_type)
        delta_tensor = torch.autograd.grad(price2, S_t2, create_graph=True)[0]
        delta_tensor.backward()
        
        gamma = S_t2.grad.item() if S_t2.grad is not None else 0.0
        
        return {
            'price': price.item(),
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def batch_price(self, S: Union[torch.Tensor, np.ndarray], 
                   K: Union[torch.Tensor, np.ndarray],
                   T: Union[torch.Tensor, np.ndarray], 
                   r: Union[torch.Tensor, np.ndarray],
                   sigma: Union[torch.Tensor, np.ndarray],
                   option_type: str = 'call') -> torch.Tensor:
        """
        Calculate option prices for batch of parameters.
        
        Args:
            S: Stock prices (batch)
            K: Strike prices (batch)
            T: Times to expiration (batch)
            r: Risk-free rates (batch)
            sigma: Volatilities (batch)
            option_type: 'call' or 'put'
            
        Returns:
            Batch of option prices
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S).float()
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float()
        if isinstance(r, np.ndarray):
            r = torch.from_numpy(r).float()
        if isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma).float()
        
        # Move to device
        S = S.to(self.device)
        K = K.to(self.device)
        T = T.to(self.device)
        r = r.to(self.device)
        sigma = sigma.to(self.device)
        
        return self.option_price(S, K, T, r, sigma, option_type)
    
    def analytical_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Greeks using analytical formulas (for validation/comparison).
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing analytical Greeks
        """
        S_t = torch.tensor(S, dtype=torch.float32)
        K_t = torch.tensor(K, dtype=torch.float32)
        T_t = torch.tensor(T, dtype=torch.float32)
        r_t = torch.tensor(r, dtype=torch.float32)
        sigma_t = torch.tensor(sigma, dtype=torch.float32)
        
        d1, d2 = self._compute_d1_d2(S_t, K_t, T_t, r_t, sigma_t)
        
        # Price
        if option_type.lower() == 'call':
            price = self.call_price(S_t, K_t, T_t, r_t, sigma_t)
            delta = self._cdf(d1)
            theta = (-(S_t * self._pdf(d1) * sigma_t) / (2 * torch.sqrt(T_t)) 
                    - r_t * K_t * torch.exp(-r_t * T_t) * self._cdf(d2))
            rho = K_t * T_t * torch.exp(-r_t * T_t) * self._cdf(d2)
        else:  # put
            price = self.put_price(S_t, K_t, T_t, r_t, sigma_t)
            delta = self._cdf(d1) - 1
            theta = (-(S_t * self._pdf(d1) * sigma_t) / (2 * torch.sqrt(T_t)) 
                    + r_t * K_t * torch.exp(-r_t * T_t) * self._cdf(-d2))
            rho = -K_t * T_t * torch.exp(-r_t * T_t) * self._cdf(-d2)
        
        # Greeks (same for call and put)
        gamma = self._pdf(d1) / (S_t * sigma_t * torch.sqrt(T_t))
        vega = S_t * self._pdf(d1) * torch.sqrt(T_t)
        
        return {
            'price': price.item(),
            'delta': delta.item(),
            'gamma': gamma.item(),
            'theta': theta.item(),
            'vega': vega.item(),
            'rho': rho.item()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Black-Scholes Model with Automatic Differentiation\n" + "="*60)
    
    # Initialize model
    bs_model = BlackScholesModel()
    
    # Example parameters
    S = 100.0   # Stock price
    K = 105.0   # Strike price
    T = 1.0     # Time to expiration (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    
    print("\nInput Parameters:")
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Expiration (T): {T} years")
    print(f"Risk-Free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    
    # Calculate call option Greeks using autograd
    print("\n" + "-"*60)
    print("CALL OPTION - Greeks via Automatic Differentiation:")
    print("-"*60)
    call_greeks_auto = bs_model.calculate_greeks(S, K, T, r, sigma, 'call')
    for greek, value in call_greeks_auto.items():
        print(f"{greek.capitalize():10s}: {value:12.6f}")
    
    # Calculate call option Greeks using analytical formulas
    print("\n" + "-"*60)
    print("CALL OPTION - Greeks via Analytical Formulas:")
    print("-"*60)
    call_greeks_analytical = bs_model.analytical_greeks(S, K, T, r, sigma, 'call')
    for greek, value in call_greeks_analytical.items():
        print(f"{greek.capitalize():10s}: {value:12.6f}")
    
    # Calculate put option Greeks
    print("\n" + "-"*60)
    print("PUT OPTION - Greeks via Automatic Differentiation:")
    print("-"*60)
    put_greeks_auto = bs_model.calculate_greeks(S, K, T, r, sigma, 'put')
    for greek, value in put_greeks_auto.items():
        print(f"{greek.capitalize():10s}: {value:12.6f}")
    
    # Batch pricing example
    print("\n" + "-"*60)
    print("Batch Pricing Example:")
    print("-"*60)
    
    # Create batch of parameters
    S_batch = torch.tensor([95.0, 100.0, 105.0, 110.0])
    K_batch = torch.tensor([100.0, 100.0, 100.0, 100.0])
    T_batch = torch.tensor([1.0, 1.0, 1.0, 1.0])
    r_batch = torch.tensor([0.05, 0.05, 0.05, 0.05])
    sigma_batch = torch.tensor([0.2, 0.2, 0.2, 0.2])
    
    call_prices = bs_model.batch_price(S_batch, K_batch, T_batch, r_batch, sigma_batch, 'call')
    
    print("\nStock Price | Call Price")
    print("-" * 30)
    for s, price in zip(S_batch, call_prices):
        print(f"${s:7.2f}     | ${price:7.4f}")
    
    print("\n" + "="*60)
    print("Testing complete!")
