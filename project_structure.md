black-scholes-nn/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
│
├── data/
│   ├── __init__.py
│   └── synthetic_data.py          # Generate synthetic Black-Scholes data
│
├── models/
│   ├── __init__.py
│   ├── black_scholes.py           # Analytical Black-Scholes formula
│   └── neural_network.py          # PyTorch NN architecture
│
├── training/
│   ├── __init__.py
│   ├── train.py                   # Training script
│   └── evaluate.py                # Model evaluation
│
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predictions.py         # Prediction endpoints
│   └── schemas.py                 # Pydantic models for request/response
│
├── frontend/
│   ├── main.py                    # Streamlit app
│   ├── components/
│   │   ├── __init__.py
│   │   ├── input_form.py          # Parameter input widgets
│   │   └── visualizations.py     # Charts and plots
│   └── utils.py                   # Helper functions for API calls
│
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebook for exploration
│
├── saved_models/
│   └── model.pth                  # Trained model weights
│
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py
│   └── test_neural_network.py
│
└── utils/
    ├── __init__.py
    └── helpers.py                 # General utility functions
