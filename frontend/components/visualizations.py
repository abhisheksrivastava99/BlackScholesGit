import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')

def plot_nn_vs_bs(S, nn_price, bs_price, option_type, figsize=(6, 4)):
    """Enhanced bar chart comparison with improved styling"""
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = ['Neural Network', 'Black-Scholes']
    values = [nn_price, bs_price]
    colors = ['#1f77b4', '#ff7f0e']  # Professional blue and orange
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'${value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel("Option Price ($)", fontsize=12, fontweight='bold')
    ax.set_title(f"Price Comparison at S = ${S}", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.15)
    
    # Enhanced styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_price_curve(x_values, nn_prices, bs_prices, option_type, x_label="Stock Price ($)", figsize=(8, 5)):
    """Enhanced line plot with improved styling"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lines with enhanced styling
    ax.plot(x_values, nn_prices, label='Neural Network', marker='o', 
            color='#1f77b4', linewidth=2.5, markersize=6, alpha=0.9)
    ax.plot(x_values, bs_prices, label='Black-Scholes', marker='s', 
            color='#ff7f0e', linewidth=2.5, markersize=5, alpha=0.9)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel("Option Price ($)", fontsize=12, fontweight='bold')
    ax.set_title(f"{option_type.capitalize()} Option Price Curves", fontsize=14, fontweight='bold', pad=20)
    
    # Enhanced legend
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    
    # Styling improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    
    fig.tight_layout()
    return fig

def plot_greeks_curve(x_values, greeks_dict, greek_name, option_type, x_label="Stock Price ($)", figsize=(7, 4)):
    """Enhanced Greeks plotting with professional styling"""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E8B57', '#DC143C', '#4169E1', '#FF6347', '#9932CC']
    
    for i, (label, values) in enumerate(greeks_dict.items()):
        ax.plot(x_values, values, label=label, linewidth=2.5, 
               color=colors[i % len(colors)], alpha=0.9)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(greek_name, fontsize=12, fontweight='bold')
    ax.set_title(f"{greek_name} Sensitivity ({option_type.capitalize()})", fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    
    # Add horizontal line at y=0 for Greeks
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    fig.tight_layout()
    return fig

def show_results_table(results_dict):
    """Enhanced table display with professional styling"""
    df = pd.DataFrame(results_dict)
    
    # Round numerical columns for better display
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        if 'Price' in col:
            df[col] = df[col].round(4)
        else:
            df[col] = df[col].round(6)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=350,
        hide_index=True
    )
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="option_pricing_results.csv",
        mime="text/csv",
        use_container_width=True
    )
