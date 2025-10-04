import streamlit as st
import requests
from components.visualizations import (
    plot_nn_vs_bs, 
    plot_price_curve, 
    plot_greeks_curve, 
    show_results_table
)
def display_sensitivity_analysis(data, option_type):
    """Function to display sensitivity analysis results"""
    # Two-column layout for curves
    curve_col1, curve_col2 = st.columns([1, 1])
    
    with curve_col1:
        st.markdown("#### 💰 Price Curves")
        fig_curve = plot_price_curve(data['S_vals'], data['nn_valid'], data['bs_valid'], option_type, figsize=(7, 5))
        st.pyplot(fig_curve, use_container_width=True)
    
    with curve_col2:
        st.markdown("#### 📊 Greeks Analysis")
        # Interactive Greek selector - this will persist now
        selected_greek = st.selectbox("Select Greek to Display:", 
                                     ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                                     key="greek_selector")
        
        greek_data_map = {
            "Delta": data['delta_valid'],
            "Gamma": data['gamma_valid'], 
            "Theta": data['theta_valid'],
            "Vega": data['vega_valid'],
            "Rho": data['rho_valid']
        }
        
        fig_greek = plot_greeks_curve(data['S_vals'], {selected_greek: greek_data_map[selected_greek]}, 
                                    selected_greek, option_type, figsize=(7, 5))
        st.pyplot(fig_greek, use_container_width=True)
    
    st.markdown("---")
    
    # Phase 3: Comprehensive Data Table
    st.markdown('<div class="section-header">📋 Comprehensive Results Table</div>', unsafe_allow_html=True)
    
    show_results_table({
        "Stock Price ($)": data['S_vals'], 
        "NN Price ($)": data['nn_valid'], 
        "BS Price ($)": data['bs_valid'],
        "Delta": data['delta_valid'], 
        "Gamma": data['gamma_valid'], 
        "Theta": data['theta_valid'],
        "Vega": data['vega_valid'], 
        "Rho": data['rho_valid']
    })

# Page configuration with professional theme
st.set_page_config(
    page_title="Black-Scholes Neural Network Dashboard", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR - Enhanced Parameter Controls
with st.sidebar:
    st.markdown("# ⚙️ Parameters")
    
    with st.container():
        st.markdown("### 📊 Market Data")
        S = st.number_input("Underlying Price ($)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        K = st.number_input("Strike Price ($)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        
        st.markdown("### ⏰ Time & Risk")
        T = st.slider("Time to Expiration (Years)", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
        r = st.slider("Risk-Free Rate", min_value=0.0, max_value=0.20, value=0.05, step=0.001, format="%.3f")
        sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
        
        st.markdown("### 📋 Option Details")
        otype = st.selectbox("Option Type", ("call", "put"), index=0)
    
    st.markdown("---")
    
    # Single action button with enhanced styling
    get_analysis = st.button("🚀 Get Complete Analysis", key="main_btn", type="primary")
    
    # Reset button to clear session state
    if st.button("🔄 Reset Analysis", key="reset_btn"):
        st.session_state.analysis_complete = False
        st.session_state.analysis_data = None
        st.rerun()
    
    st.markdown("---")
    
    # Info panel
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Neural Black-Scholes Dashboard**
        
        This dashboard provides:
        - 🎯 Single point predictions
        - 📈 Price curve analysis  
        - 📊 Greeks sensitivity analysis
        - 📋 Comprehensive data table
        
        **Created by:** Abhishek Srivastava  
        **Institution:** NTU Singapore  
        **Program:** FinTech Masters
        """)

params = {
    "S": float(S), "K": float(K), "T": float(T), 
    "r": float(r), "sigma": float(sigma), "option_type": otype
}

# MAIN DASHBOARD
st.markdown('<h1 class="main-header">🚀 Neural Black-Scholes Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Option Pricing with Machine Learning</p>', unsafe_allow_html=True)

# Quick metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Underlying Price", f"${S:.2f}")
with col2:
    st.metric("Strike Price", f"${K:.2f}")
with col3:
    st.metric("Time to Expiry", f"{T:.2f} years")
with col4:
    moneyness = S/K
    st.metric("Moneyness", f"{moneyness:.3f}", delta=f"{'ITM' if (moneyness > 1 and otype == 'call') or (moneyness < 1 and otype == 'put') else 'OTM'}")

st.markdown("---")

# MAIN ANALYSIS SECTION
if get_analysis:
    # Reset analysis state when new analysis is requested
    st.session_state.analysis_complete = False
    
    # Phase 1: Single Point Analysis
    st.markdown('<div class="section-header">🎯 Single Point Analysis</div>', unsafe_allow_html=True)
    
    try:
        # Get predictions with loading indicators
        with st.spinner("Getting Neural Network prediction..."):
            nn_response = requests.post("http://localhost:8000/predict/", json=params, timeout=10)
            nn_response.raise_for_status()
            nn_data = nn_response.json()
        
        with st.spinner("Calculating Black-Scholes and Greeks..."):
            bs_response = requests.post("http://localhost:8000/predict/greeks", json=params, timeout=10)
            bs_response.raise_for_status()
            bs_data = bs_response.json()
        
        # Single Point Results Layout
        point_col1, point_col2 = st.columns([1, 1])
        
        with point_col1:
            # Price comparison metrics
            st.markdown("#### 💰 Price Comparison")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("🤖 NN Price", f"${nn_data['option_price']:.4f}")
            with metric_col2:
                st.metric("📐 BS Price", f"${bs_data['price']:.4f}")
            with metric_col3:
                error = abs(nn_data['option_price'] - bs_data['price'])
                st.metric("❌ Error", f"${error:.4f}")
            
            # Greeks metrics
            st.markdown("#### 📊 The Greeks")
            greeks_col1, greeks_col2 = st.columns(2)
            
            with greeks_col1:
                st.metric("Delta (Δ)", f"{bs_data['delta']:.4f}")
                st.metric("Gamma (Γ)", f"{bs_data['gamma']:.4f}")
                st.metric("Theta (Θ)", f"{bs_data['theta']:.4f}")
            
            with greeks_col2:
                st.metric("Vega (ν)", f"{bs_data['vega']:.4f}")
                st.metric("Rho (ρ)", f"{bs_data['rho']:.4f}")
        
        with point_col2:
            # Price comparison chart
            st.markdown("#### 📊 Visual Comparison")
            fig_comparison = plot_nn_vs_bs(S, nn_data['option_price'], bs_data['price'], otype, figsize=(7, 5))
            st.pyplot(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        
        # Phase 2: Sensitivity Analysis
        st.markdown('<div class="section-header">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner("Performing comprehensive sensitivity analysis..."):
            S_values = list(range(50, 151, 5))
            nn_prices, bs_prices = [], []
            delta_list, gamma_list, theta_list, vega_list, rho_list = [], [], [], [], []
            
            # Progress bar for user feedback
            progress_bar = st.progress(0)
            
            for i, S_sweep in enumerate(S_values):
                sweep_params = params.copy()
                sweep_params["S"] = S_sweep
                
                try:
                    nn_resp = requests.post("http://localhost:8000/predict/", json=sweep_params, timeout=5)
                    nn_resp.raise_for_status()
                    nn_prices.append(nn_resp.json()['option_price'])
                except:
                    nn_prices.append(None)
                
                try:
                    gr_resp = requests.post("http://localhost:8000/predict/greeks", json=sweep_params, timeout=5)
                    gr_resp.raise_for_status()
                    g = gr_resp.json()
                    bs_prices.append(g["price"])
                    delta_list.append(g["delta"])
                    gamma_list.append(g["gamma"])
                    theta_list.append(g["theta"])
                    vega_list.append(g["vega"])
                    rho_list.append(g["rho"])
                except:
                    bs_prices.append(None)
                    delta_list.append(None)
                    gamma_list.append(None)
                    theta_list.append(None)
                    vega_list.append(None)
                    rho_list.append(None)
                
                progress_bar.progress((i + 1) / len(S_values))
            
            # Filter valid data
            valid = [i for i, p in enumerate(nn_prices) if p is not None and bs_prices[i] is not None]
            
            if valid:
                S_vals = [S_values[i] for i in valid]
                nn_valid = [nn_prices[i] for i in valid]
                bs_valid = [bs_prices[i] for i in valid]
                delta_valid = [delta_list[i] for i in valid]
                gamma_valid = [gamma_list[i] for i in valid]
                theta_valid = [theta_list[i] for i in valid]
                vega_valid = [vega_list[i] for i in valid]
                rho_valid = [rho_list[i] for i in valid]
                
                # Store data in session state
                st.session_state.analysis_data = {
                    'S_vals': S_vals,
                    'nn_valid': nn_valid,
                    'bs_valid': bs_valid,
                    'delta_valid': delta_valid,
                    'gamma_valid': gamma_valid,
                    'theta_valid': theta_valid,
                    'vega_valid': vega_valid,
                    'rho_valid': rho_valid,
                    'params': params.copy()
                }
                st.session_state.analysis_complete = True
                
                # Display results
                display_sensitivity_analysis(st.session_state.analysis_data, otype)
                
                st.success("✅ Complete analysis generated successfully!")
            else:
                st.error("❌ No valid data points obtained from sensitivity analysis")
    
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        st.info("💡 Make sure your backend server is running on http://localhost:8000")

elif st.session_state.analysis_complete and st.session_state.analysis_data:
    # Display previously generated analysis
    st.markdown('<div class="section-header">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
    display_sensitivity_analysis(st.session_state.analysis_data, otype)

else:
    # Welcome screen when no button is pressed
    st.markdown("### 👋 Welcome to the Neural Black-Scholes Dashboard")
    st.info("📋 **Instructions:**\n1. Set your option parameters in the sidebar\n2. Click '🚀 Get Complete Analysis' to see all results\n3. The analysis includes single point predictions, sensitivity curves, Greeks analysis, and a comprehensive data table")
    
    # Show example parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 Current Parameters:**")
        st.write(f"- Underlying: ${S}")
        st.write(f"- Strike: ${K}")
        st.write(f"- Time: {T} years")
    with col2:
        st.markdown("**⚙️ Risk Parameters:**")
        st.write(f"- Risk-free rate: {r:.3f}")
        st.write(f"- Volatility: {sigma:.3f}")
        st.write(f"- Option type: {otype.capitalize()}")
    with col3:
        st.markdown("**🎯 Analysis Includes:**")
        st.write("- Price comparison")
        st.write("- Sensitivity curves") 
        st.write("- Greeks analysis")
        st.write("- Data export")

def display_sensitivity_analysis(data, option_type):
    """Function to display sensitivity analysis results"""
    # Two-column layout for curves
    curve_col1, curve_col2 = st.columns([1, 1])
    
    with curve_col1:
        st.markdown("#### 💰 Price Curves")
        fig_curve = plot_price_curve(data['S_vals'], data['nn_valid'], data['bs_valid'], option_type, figsize=(7, 5))
        st.pyplot(fig_curve, use_container_width=True)
    
    with curve_col2:
        st.markdown("#### 📊 Greeks Analysis")
        # Interactive Greek selector - this will persist now
        selected_greek = st.selectbox("Select Greek to Display:", 
                                     ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                                     key="greek_selector")
        
        greek_data_map = {
            "Delta": data['delta_valid'],
            "Gamma": data['gamma_valid'], 
            "Theta": data['theta_valid'],
            "Vega": data['vega_valid'],
            "Rho": data['rho_valid']
        }
        
        fig_greek = plot_greeks_curve(data['S_vals'], {selected_greek: greek_data_map[selected_greek]}, 
                                    selected_greek, option_type, figsize=(7, 5))
        st.pyplot(fig_greek, use_container_width=True)
    
    st.markdown("---")
    
    # Phase 3: Comprehensive Data Table
    st.markdown('<div class="section-header">📋 Comprehensive Results Table</div>', unsafe_allow_html=True)
    
    show_results_table({
        "Stock Price ($)": data['S_vals'], 
        "NN Price ($)": data['nn_valid'], 
        "BS Price ($)": data['bs_valid'],
        "Delta": data['delta_valid'], 
        "Gamma": data['gamma_valid'], 
        "Theta": data['theta_valid'],
        "Vega": data['vega_valid'], 
        "Rho": data['rho_valid']
    })
