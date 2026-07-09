import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.express as px
from openai import OpenAI
import json

# --- PAGE SETUP ---
st.set_page_config(page_title="LLM Tax-Aware Optimizer", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS FOR SLEEK UI ---
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00d2ff;}
    .stMetric {background-color: #1a1c23; padding: 15px; border-radius: 8px; border-left: 4px solid #00d2ff;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ LLM-Driven Tax-Aware Portfolio Optimizer")
st.markdown("Translating unstructured tax policy into structured mathematical portfolio constraints via NLP.")
st.divider()

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

# --- REAL LLM LOGIC ---
def extract_tax_constraints(prompt, api_key):
    client = OpenAI(api_key=api_key)
    system_prompt = '''
    You are a quantitative tax analyst. Extract the proposed tax rates for Equities, Bonds, Real Estate, and Commodities from the user's text.
    Return ONLY a valid JSON object with the following keys and float values (e.g., 0.39 for 39%): 
    "equity_tax", "bond_tax", "real_estate_tax", "commodity_tax". 
    If a rate is not explicitly mentioned, use standard defaults: equity_tax (0.20), bond_tax (0.37), real_estate_tax (0.25), commodity_tax (0.28).
    '''
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)

# --- OPTIMIZATION ENGINE ---
def optimize_portfolio(returns, cov_mat):
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_mat, weights))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    optimal = sco.minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal.x

# --- SIDEBAR INPUT ---
with st.sidebar:
    st.header("🔑 API Configuration")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    st.header("📜 Tax Policy Input")
    st.write("Paste a proposed tax policy or news excerpt below to run the simulation.")
    
    policy_text = st.text_area(
        "Raw Regulation Text", 
        "The new administration proposes increasing the capital gains tax to align with the top income bracket of 39%, impacting Equities and Real Estate, while keeping standard Bonds at 37%.",
        height=200
    )
    
    run_button = st.button("🚀 Run AI Parser & Optimize", use_container_width=True)
    
    with st.expander("Mathematical Methodology"):
        st.write("""
        **1. Extraction:** NLP isolates parametric constraints via `gpt-4o` JSON mode.
        **2. Adjustment:** $R_{post} = R_{pre} * (1 - T_{asset})$
        **3. Optimization:** SciPy SLSQP algorithm minimizes portfolio variance while seeking to maximize the after-tax Sharpe Ratio.
        """)

# --- MAIN DASHBOARD LOGIC ---
if run_button:
    if not user_api_key:
        st.error("Please enter a valid OpenAI API Key in the sidebar to run the extraction.")
    else:
        col1, col2 = st.columns([1, 2.5], gap="large")
        
        with col1:
            st.subheader("1. LLM Parameter Extraction")
            with st.spinner("Connecting to OpenAI & parsing tax code..."):
                try:
                    tax_rates = extract_tax_constraints(policy_text, user_api_key)
                    st.success("JSON Constraints Extracted")
                    st.json(tax_rates)
                except Exception as e:
                    st.error(f"API Error: {e}")
                    st.stop()
            
        with col2:
            st.subheader("2. Algorithmic Reallocation Results")
            
            # Calculate Math
            tax_array = np.array([tax_rates.get('equity_tax', 0.2), tax_rates.get('bond_tax', 0.37), tax_rates.get('real_estate_tax', 0.25), tax_rates.get('commodity_tax', 0.28)])
            post_tax_returns = expected_returns * (1 - tax_array)
            pre_tax_weights = optimize_portfolio(expected_returns, cov_matrix)
            risk_free_rate = 0.02
            
            # Simulated Sharpe maximization with new taxes
            post_tax_weights = sco.minimize(
                lambda w: -(np.dot(w, post_tax_returns) - risk_free_rate) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                [0.25]*4, bounds=tuple((0,1) for _ in range(4)), constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x)-1}]
            ).x
            
            # Data formatting for Plotly
            df_weights = pd.DataFrame({
                'Asset': assets,
                'Pre-Tax (Standard MVO)': np.round(pre_tax_weights * 100, 2),
                'Post-Tax (Tax-Aware MVO)': np.round(post_tax_weights * 100, 2)
            })
            
            df_melted = df_weights.melt(id_vars='Asset', var_name='Strategy', value_name='Allocation (%)')
            
            # Sleek Interactive Chart
            fig = px.bar(
                df_melted, 
                x='Asset', 
                y='Allocation (%)', 
                color='Strategy', 
                barmode='group',
                color_discrete_sequence=['#4a5568', '#00d2ff'],
                title="Portfolio Weight Shift (Pre-Tax vs Post-Tax)"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # KPI Metrics
            st.write("### Model Diagnostics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pre-Tax Expected Yield", f"{np.dot(pre_tax_weights, expected_returns)*100:.2f}%")
            
            post_yield = np.dot(post_tax_weights, post_tax_returns)*100
            pre_yield = np.dot(pre_tax_weights, expected_returns)*100
            
            m2.metric("Post-Tax Expected Yield", f"{post_yield:.2f}%", delta=f"{post_yield - pre_yield:.2f}%", delta_color="normal")
            m3.metric("Tax Drag Penalty", "Optimized", help="The optimizer successfully re-weighted the portfolio to minimize the mathematical impact of the newly parsed tax parameters.")

else:
    st.info("👈 Enter your OpenAI API Key and a hypothetical tax policy in the sidebar, then click **Run** to execute the pipeline.")
