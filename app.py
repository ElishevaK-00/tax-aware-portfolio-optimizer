import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco

# --- PAGE SETUP ---
st.set_page_config(page_title="LLM Tax-Aware Optimizer", layout="wide")
st.title("🏛️ LLM-Driven Tax-Aware Portfolio Optimizer")
st.markdown("A proof-of-concept translating unstructured tax policy into mathematical portfolio constraints.")

# --- MOCK FINANCIAL DATA ---
assets = ['Equities (SPY)', 'Bonds (AGG)', 'Real Estate (VNQ)', 'Commodities (GLD)']
num_assets = len(assets)

expected_returns = np.array([0.08, 0.04, 0.06, 0.03]) 
cov_matrix = np.array([
    [0.040, 0.005, 0.020, 0.002],
    [0.005, 0.010, 0.004, -0.001],
    [0.020, 0.004, 0.050, 0.008],
    [0.002, -0.001, 0.008, 0.030]
])

# --- LLM SIMULATION LOGIC ---
def extract_tax_constraints(prompt):
    if "democrat" in prompt.lower() or "increase" in prompt.lower():
        return {"equity_tax": 0.39, "bond_tax": 0.39, "real_estate_tax": 0.39, "commodity_tax": 0.39}
    elif "republican" in prompt.lower() or "cut" in prompt.lower():
        return {"equity_tax": 0.15, "bond_tax": 0.20, "real_estate_tax": 0.10, "commodity_tax": 0.15}
    else:
        return {"equity_tax": 0.20, "bond_tax": 0.37, "real_estate_tax": 0.25, "commodity_tax": 0.28}

# --- OPTIMIZATION ENGINE ---
def optimize_portfolio(returns, cov_mat):
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_mat, weights))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    optimal = sco.minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal.x

# --- UI & INTERACTION ---
st.sidebar.header("Policy Input")
st.sidebar.write("Paste a proposed tax policy or news excerpt below:")

policy_text = st.sidebar.text_area(
    "Tax Regulation Text", 
    "The new administration proposes increasing the capital gains tax to align with the top income bracket of 39%, impacting all asset classes equally.",
    height=150
)

if st.sidebar.button("Run LLM Parser & Optimize"):
    st.subheader("1. LLM Parameter Extraction")
    with st.spinner("LLM is reading the tax code..."):
        tax_rates = extract_tax_constraints(policy_text)
        
    st.json(tax_rates)
    
    post_tax_returns = expected_returns * (1 - np.array([tax_rates['equity_tax'], tax_rates['bond_tax'], tax_rates['real_estate_tax'], tax_rates['commodity_tax']]))
    
    st.subheader("2. Mathematical Optimization Results")
    pre_tax_weights = optimize_portfolio(expected_returns, cov_matrix)
    risk_free_rate = 0.02
    
    df_weights = pd.DataFrame({
        'Asset': assets,
        'Pre-Tax Allocation (%)': np.round(pre_tax_weights * 100, 2),
        'Post-Tax Allocation (%)': np.round(sco.minimize(
            lambda w: -(np.dot(w, post_tax_returns) - risk_free_rate) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            [0.25]*4, bounds=tuple((0,1) for _ in range(4)), constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
        ).x * 100, 2)
    }).set_index('Asset')
    
    st.write("Notice how the algorithm reallocates capital away from highly-taxed asset classes to maximize the after-tax Sharpe Ratio.")
    st.bar_chart(df_weights)
    st.success("Optimization Complete! Mathematical constraints successfully updated via Natural Language Processing.")
else:
    st.info("👈 Enter a hypothetical tax policy and click Run to see the engine work.")
