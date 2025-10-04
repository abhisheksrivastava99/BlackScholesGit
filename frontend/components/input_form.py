"""
frontend/components/input_form.py
Reusable Streamlit option parameter input form.
"""

import streamlit as st

def option_input_form():
    """Streamlit UI for Black-Scholes option parameters."""
    st.header("Option Parameters")
    S = st.number_input("Stock/Underlying Price (S)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    T = st.slider("Time to Expiration (Years)", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
    r = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=0.10, value=0.05, step=0.001)
    sigma = st.slider("Volatility (%)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    option_type = st.selectbox("Option Type", ("call", "put"))
    
    # Convert r and sigma from percent to decimal
    r = float(r)
    sigma = float(sigma)
    
    # Return as dict for easy API usage
    return {
        "S": float(S),
        "K": float(K),
        "T": float(T),
        "r": r,
        "sigma": sigma,
        "option_type": option_type
    }
