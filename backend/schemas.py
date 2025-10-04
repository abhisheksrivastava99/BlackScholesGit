"""
backend/schemas.py
Pydantic data models for FastAPI input and output.
"""

from pydantic import BaseModel, Field, condecimal
from typing import Optional, Literal, Dict

class OptionParams(BaseModel):
    """Input parameters for option pricing."""
    S: float = Field(..., gt=0, description="Stock/Underlying price (must be positive)")
    K: float = Field(..., gt=0, description="Strike price (must be positive)")
    T: float = Field(..., gt=0, description="Time to expiration in years (must be positive, e.g., 0.5 for 6 months)")
    r: float = Field(..., ge=0, description="Risk-free annual interest rate (as decimal, e.g. 0.05 for 5%)")
    sigma: float = Field(..., gt=0, description="Annualized volatility (as decimal, e.g. 0.2 for 20%)")
    option_type: Literal['call', 'put'] = Field(..., description="Option type: 'call' or 'put'")

class OptionPrediction(BaseModel):
    """Output for option price prediction."""
    option_price: float = Field(..., description="Predicted option price")

class OptionGreeks(BaseModel):
    """Output for option Greeks (optional, advanced)."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class ErrorResponse(BaseModel):
    """Model for error messages (optional)."""
    detail: str

