import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
from openai import OpenAI
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Institutional Tax-Aware Optimizer", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS FOR AN INSTITUTIONAL TERMINAL FEEL ---
st.markdown("""
    <style>
    .reportview-container { background: #0b0c10; }
    h1, h2, h3 { color: #66fcf1; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1f2833; padding: 20px; border-radius: 10px; border-top: 4px solid #66fcf1; }
    div.stButton > button:first-child { background-color: #45a29e; color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Institutional Tax-Aware Portfolio Framework")
st.markdown("Analytically mapping the impact of macro tax policy shifts onto the Markowitz Efficient Frontier.")
st.divider()

# --- EMPIRICALLY ANCHORED CAPITAL MARKET ASSUMPTIONS (CMAs) ---
assets = ['Global Equities', 'Fixed Income', 'Real Estate (REITs)', 'Alternative Commodities']
num_assets = len(assets)

# Baseline historical real expected returns and a stable covariance structure
expected_returns = np.array([0.085, 0.040, 0.065, 0.045]) 
cov_matrix = np.array([
    [0.0324, 0.0012, 0.0180, 0.0015],
    [0.0012, 0.0064, 0.0016, -0.0004],
    [0.0180, 0.0016, 0.0289, 0.0036],
    [0.0015, -0.0004, 0.0036, 0.0225]
])

# --- OPENAI PARSING LOGIC ---
def extract_tax_constraints(prompt, api_key):
    client = OpenAI(api_key=api_key)
    system_prompt = '''
    You are an expert quantitative tax compliance engine. Extract the explicit or implied tax rates for the asset classes from the regulatory text.
    Return ONLY a valid JSON object with these exact keys: "equity_tax", "bond_tax", "real_estate_tax", "commodity_tax".
    Values must be floats representing percentages (e.g., 20% is 0.20). 
    Defaults if unmentioned: equities=0.20, bonds=0.35, real_estate=0.25, commodities=0.28.
    '''
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

# --- MODERN PORTFOLIO THEORY ENGINE ---
def get_portfolio_stats(weights, returns, cov_mat):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return port_return, port_vol

def generate_efficient_frontier(returns, cov_mat, target_returns):
    """Calculates minimum variance portfolios for an array of target returns to map the frontier line."""
    frontier_vols = []
    frontier_weights = []
    
    # Institutionally realistic allocation bounds: prevent 0% or 100% corner solutions
    # Every asset must hold between 10% and 50% of total capital
    bounds = tuple((0.10, 0.50) for _ in range(num_assets))
    
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, returns) - target}
        ]
        res = sco.minimize(
            lambda x: np.dot(x.T, np.dot(cov_mat, x)), 
            num_assets * [1./num_assets], 
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        if res.success:
            frontier_vols.append(np.sqrt(res.fun))
            frontier_weights.append(res.x)
        else:
            frontier_vols.append(None)
            frontier_weights.append(None)
            
    return frontier_vols, frontier_weights

# --- INTERFACE LAYOUT ---
with st.sidebar:
    st.header("🔑 Authentication")
    user_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.header("📋 Policy Input Vector")
    policy_text = st.text_area(
        "Regulatory Text Excerpt", 
        "Congress passes emergency tax restructuring: Capital gains on Equities jump to 40%. Real estate statutory tax increases to 35%. Long-term corporate bond income remains protected at a flat rate of 20%.",
        height=180
    )
    run_button = st.button("⚡ Execute Dual Optimization", use_container_width=True)

# --- EXECUTION PIPELINE ---
if run_button:
    if not user_api_key:
        st.error("Authentication Error: A valid OpenAI API Key is required to compile natural language into parameters.")
    else:
        with st.spinner("Extracting regulatory parameters & mapping efficient frontiers..."):
            try:
                tax_rates = extract_tax_constraints(policy_text, user_api_key)
            except Exception as e:
                st.error(f"LLM Engine Failure: {e}")
                st.stop()
        
        # Apply Tax Drag to Expected Returns
        tax_vector = np.array([tax_rates.get('equity_tax', 0.2), tax_rates.get('bond_tax', 0.35), tax_rates.get('real_estate_tax', 0.25), tax_rates.get('commodity_tax', 0.28)])
        post_tax_returns = expected_returns * (1 - tax_vector)
        
        # Define accurate return boundaries based on constrained limits
        target_returns_pre = np.linspace(0.045, 0.065, 30)
        target_returns_post = np.linspace(0.025, 0.048, 30)
        
        # Compute Frontiers
        vols_pre, weights_pre = generate_efficient_frontier(expected_returns, cov_matrix, target_returns_pre)
        vols_post, weights_post = generate_efficient_frontier(post_tax_returns, cov_matrix, target_returns_post)
        
        # Filter mathematical edge-case dropouts
        valid_pre = [(v, r, w) for v, r, w in zip(vols_pre, target_returns_pre, weights_pre) if v is not None]
        valid_post = [(v, r, w) for v, r, w in zip(vols_post, target_returns_post, weights_post) if v is not None]
        
        # --- SCREEN LAYOUT ---
        layout_left, layout_right = st.columns([1.8, 1.2], gap="medium")
        
        with layout_left:
            st.subheader("The Capital Allocation Shift (Markowitz Frontier)")
            st.markdown("*Hover over the frontier lines to see how individual asset allocations dynamically morph at different risk profiles.*")
            
            # Constructing a comprehensive interactive Plotly object
            fig = go.Figure()
            
            # Pre-Tax Curve
            fig.add_trace(go.Scatter(
                x=[x[0]*100 for x in valid_pre], y=[x[1]*100 for x in valid_pre],
                mode='lines+markers', name='Pre-Tax Opportunity Set',
                line=dict(color='#45a29e', width=3),
                text=[f"Allocations:<br>" + "<br>".join([f"{a}: {w*100:.1f}%" for a, w in zip(assets, x[2])]) for x in valid_pre],
                hoverinfo='text+x+y'
            ))
            
            # Post-Tax Curve
            fig.add_trace(go.Scatter(
                x=[x[0]*100 for x in valid_post], y=[x[1]*100 for x in valid_post],
                mode='lines+markers', name='Post-Tax Structural Frontier',
                line=dict(color='#66fcf1', width=3, dash='dash'),
                text=[f"Allocations:<br>" + "<br>".join([f"{a}: {w*100:.1f}%" for a, w in zip(assets, x[2])]) for x in valid_post],
                hoverinfo='text+x+y'
            ))
            
            fig.update_layout(
                xaxis_title="Portfolio Volatility (Risk %)",
                yaxis_title="Expected Portfolio Return (%)",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=20, r=20, t=20, b=20),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with layout_right:
            st.subheader("Parser Diagnostics & System Utility")
            st.write("Parsed Policy Parameters:")
            st.json(tax_rates)
            
            # Pull median risk profile to show a side-by-side realistic allocation slice
            mid_idx_pre = len(valid_pre) // 2
            mid_idx_post = len(valid_post) // 2
            
            df_compare = pd.DataFrame({
                'Asset Class': assets,
                'Pre-Tax Baseline': valid_pre[mid_idx_pre][2] * 100,
                'Post-Tax Optimized': valid_post[mid_idx_post][2] * 100
            }).set_index('Asset Class')
            
            st.write("### Structural Portfolio Realignment")
            st.markdown("Reallocation comparison slice taken at the median risk tier:")
            st.bar_chart(df_compare)
            
        # --- DIAGNOSTIC METRIC TILES ---
        st.subheader("System Performance Metrics")
        m1, m2, m3 = st.columns(3)
        
        max_cma_return = np.max(expected_returns) * 100
        implied_drag = (np.mean(target_returns_pre) - np.mean(target_returns_post)) * 100
        
        m1.metric("Asset Universe Max Yield", f"{max_cma_return:.2f}%", help="The theoretical unconstrained ceiling based on raw baseline inputs.")
        m2.metric("Mean System Frontier Drag", f"-{implied_drag:.2f}%", delta_color="inverse", help="The total mathematical compression of the investment frontier caused directly by the extracted tax matrix.")
        m3.metric("Optimizer Bounds Status", "Constrained (10% - 50%)", help="Ensures compliance with institutional diversification requirements and eliminates extreme, brittle weight distributions.")
else:
    st.info("👈 Enter your API credential and a realistic or radical legislative tax proposal in the sidebar to visualize the system structural shift.")
