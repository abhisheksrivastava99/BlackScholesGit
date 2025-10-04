"""
backend/routers/predictions.py
Prediction endpoints for neural Black-Scholes FastAPI backend.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict
import torch
import os

from schemas import OptionParams, OptionPrediction, OptionGreeks
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the model, normalizer, and Black-Scholes code
from models.neural_network import OptionPricingNN, InputNormalizer
from models.black_scholes import BlackScholesModel

# Define constants for saved model paths
MODEL_PATH = os.path.join("saved_models", "best_model.pth")
NORMALIZER_PATH = os.path.join("saved_models", "normalizer.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate router
router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    responses={404: {"description": "Not found"}},
)

# Load model and normalizer once at startup
model = OptionPricingNN.load_model(MODEL_PATH, device=device)
model.eval()
normalizer = InputNormalizer()
normalizer.load(NORMALIZER_PATH)
bs_model = BlackScholesModel(device=device)


@router.post("/", response_model=OptionPrediction)
def predict_option_price(params: OptionParams) -> OptionPrediction:
    """
    Predict option price with trained neural network.
    """
    try:
        # Convert incoming data to tensor shape (1, 5)
        # Convert option_type ('call'/'put') to int: 0 for call, 1 for put
        option_type_num = 0 if params.option_type == "call" else 1
        # Prepare the features tensor with 6 features
        features = torch.tensor([[params.S, params.K, params.T, params.r, params.sigma, option_type_num]], dtype=torch.float32)
        features_norm = normalizer.transform(features)
        features_norm = features_norm.to(device)
        with torch.no_grad():
            output = model(features_norm).cpu().numpy().flatten()[0]
        return OptionPrediction(option_price=float(output))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/greeks", response_model=OptionGreeks)
def predict_option_with_greeks(params: OptionParams) -> OptionGreeks:
    """
    Predict Black-Scholes price and Greeks (using analytical formulas).
    """
    try:
        greeks = bs_model.analytical_greeks(
            S=params.S,
            K=params.K,
            T=params.T,
            r=params.r,
            sigma=params.sigma,
            option_type=params.option_type,
        )
        return OptionGreeks(**greeks)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
