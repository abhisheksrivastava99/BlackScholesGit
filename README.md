# Neural Black-Scholes Dashboard

**A modern machine learning and API-powered solution for fast and accurate option pricing using the Black-Scholes model, with interactive Streamlit and FastAPI deployments.**

## Table of Contents
- Project Overview
- Features
- Architecture
- Installation & Setup
- Usage Guide
    - Streamlit Cloud App
    - Backend API (FastAPI)
    - Example Inputs for Real-World Testing
- Model Design & Training
- Evaluation & Metrics
- Project Structure
- Advanced Usage
- References

***

## Project Overview

This project provides an end-to-end pipeline for **estimating European option prices** using a neural network model trained on Black-Scholes analytical data. It features:
- **API backend** powered by FastAPI (deployed on Render)
- **Interactive web frontend** built with Streamlit (deployed on Streamlit Community Cloud)
- Modular Python package for training, evaluation, and extension

***

## Features

- **Neural Network approximator** for option prices (PyTorch, fully differentiable, suitable for Greeks)
- **Synthetic data generation** using Black-Scholes formulas
- **Analytical and model-based Greeks computation**
- **Batch evaluation and regression metrics**
- **Comprehensive training and evaluation pipeline**
- **Modern web UI for user-friendly interaction**
- **Cloud deployments for both backend and frontend**
- **FastAPI REST API for programmatic access**

***

## Architecture

### Overview Diagram

```
[Streamlit Frontend] <----> [FastAPI Backend/API] <----> [Model Core]
                                    |
                         [PyTorch Neural Network]
                                    |
                       [Synthetic Data Generator]
                                    |
                    [Black-Scholes Analytical Engine]
```

### Module Breakdown

- **Streamlit App**
    - Deploys the UI for option pricing, model prediction, and analytics.
    - Connects (REST calls) to FastAPI backend.

- **FastAPI Backend**
    - Serves prediction endpoints for option prices and Greeks.
    - Accepts parameterized input (S, K, T, r, sigma, type).

- **Model Core (`neural_network.py`, `black_scholes.py`)**
    - PyTorch neural networks for approximating pricing surface.
    - Analytical engine for exact Black-Scholes prices and Greeks.
    - Model persistence and normalization utilities.

- **Training Module (`train.py`)**
    - Synthetic data creation and normalization.
    - Early stopping, LR scheduling, overfitting diagnostics.
    - Logging, checkpoints, and reproducibility.

- **Evaluation Module (`evaluate.py`)**
    - Batch evaluation against analytical Black-Scholes.
    - Metrics: MSE, MAE, MAPE, RMSE, R2, Max Error.
    - Detailed visualizations and CSV report.

- **Synthetic Data Generator (`synthetic_data.py`)**
    - Generates diverse input parameters covering real market conditions.
    - Supports noise addition for robustness.

***

## Installation & Setup

**Dependencies**
- Python 3.8+
- PyTorch, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Scikit-learn
- FastAPI, Uvicorn, Streamlit

Install requirements:
```bash
pip install -r requirements.txt
```

***

## Usage Guide

### Streamlit Cloud App

**Try it online:**  
- [Streamlit Frontend](https://blackscholesestimator.streamlit.app/):  
  - Enter your option parameters:  
    - Asset Price ($$S$$), Strike Price ($$K$$), Time to Expiry ($$T$$), Risk-Free Rate ($$r$$), Volatility ($$\sigma$$), Option Type (*call/put*)

**Example Inputs to Test:**
- S: 100, K: 105, T: 1.0, r: 0.05, sigma: 0.2, Type: call
- S: 120, K: 110, T: 0.5, r: 0.03, sigma: 0.15, Type: put

### Backend FastAPI

API endpoint: [https://blackscholesgit.onrender.com](https://blackscholesgit.onrender.com)

Sample REST call:
```bash
curl -X POST "https://blackscholesgit.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"S": 100, "K": 105, "T": 1.0, "r": 0.05, "sigma": 0.2, "type": "call"}'
```

***

## Model Design & Training

**Neural Network Architectures:**
- Standard and deep architectures (configurable layers, ELU activation for gradient smoothness)
- Input: [S, K, T, r, sigma, option_type]
- Output: Option price (regression)
- BatchNorm, Dropout regularization

**Training Pipeline:**
- Synthetic dataset creation covering these parameter ranges:
    - S: 50–150
    - K: 50–150
    - T: 0.1–2.0 years
    - r: 0.01–0.1 (annual)
    - σ: 0.1–0.5 (volatility)
    - Option Type: call/put
- Early stopping, LR scheduling, reproducibility via seed
- Normalization for feature scaling

***

## Evaluation & Metrics

- **Metrics:** Mean Squared Error, RMSE, MAE, MAPE, R², Median Error, Max Error.
- **Comprehensive reports:** CSV and visualizations (actual vs predicted, error histograms, comparison with analytical prices).
- **Comparison:** Neural network predictions are benchmarked against Black-Scholes analytical calculations.

***

## Project Structure

```text
project-root/
├── models/
│   ├── neural_network.py      # PyTorch NN architectures, normalizers
│   └── black_scholes.py      # Analytical model, Greeks computation
├── data/
│   └── synthetic_data.py     # Synthetic data generation
├── training/
│   ├── train.py              # Training script, pipeline
│   └── evaluate.py           # Evaluation, benchmarking
├── frontend/
│   └── streamlit_app.py      # Streamlit UI implementation
├── backend/
│   └── fastapi_app.py        # FastAPI app setup
├── savedmodels/              # Model checkpoints and artifacts
├── requirements.txt
└── README.md                 # Project documentation
```

***

## Advanced Usage & Extensibility

- **Custom parameter ranges**: User may select custom market ranges for S, K, T, r, σ.
- **Put/Call switch**: Supports both call and put European options.
- **Model retraining**: Replace synthetic data with real market data for improved realism.
- **Greeks calculation**: Full support for Delta, Gamma, Theta, Vega, Rho via autograd and formulas.

***

## References

- Black-Scholes Model: Wikipedia, Investopedia
- PyTorch Documentation
- FastAPI and Streamlit Docs

***

## Notes and Recommendations

- **Testing:** Use realistic market conditions for S, K, T, r, σ based on current option contracts. Try boundary values and extreme/edge cases.
- **Deployment Checks:** Ensure both backend (Render) and frontend (Streamlit Cloud) are live before testing complete workflow.
- **Performance:** For large batch prediction, API endpoints support batch inputs.
- **Extending:** Modify synthetic_data.py to incorporate dividends or American options for future work.

***

**Contact**  
For questions or contributions, please create issues or pull requests on the [GitHub repo](https://github.com/abhisheksrivastava99/BlackScholesGit/).

***

This README provides a robust professional summary for your project, fully documenting its intent, implementation, interfaces, model, and practical user instructions. For further polish, add technical diagrams and usage GIFs/screenshots of the deployed apps.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/91201796/ad13f953-52bf-4dc8-a601-e4e05ccc16b4/neural_network.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/91201796/087337b6-db29-4f32-9ae2-c7859a6e1a75/train.py)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/91201796/433af580-28a0-4420-bdf6-fa102e5c2910/evaluate.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/91201796/5dd59cfd-9825-4970-8511-3364c4596381/synthetic_data.py)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/91201796/d50f38a8-f8cd-4b38-bea0-0d08817c77a4/black_scholes.py)
