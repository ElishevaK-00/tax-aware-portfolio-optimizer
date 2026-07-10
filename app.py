import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Quantitative Tax Terminal", page_icon="📈", layout="wide")

# --- CUSTOM CSS FOR DARK INSTITUTIONAL THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; }
    h1, h2, h3, h4 { color: #63b3ed; font-family: 'Inter', sans-serif; }
    .metric-container { background-color: #1a202c; padding: 20px; border-radius: 10px; border-left: 4px solid #63b3ed; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-label { color: #a0aec0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ffffff; font-size: 28px; font-weight: bold; }
    .metric-sub { color: #fc8181; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Institutional Tax-Aware Portfolio Terminal")
st.markdown("Dynamic visualization of macro tax policy impacts on the Markowitz Efficient Frontier.")
st.divider()

# --- EMPIRICALLY ANCHORED DATA ---
assets = ['Global Equities', 'Fixed Income', 'Real Estate', 'Commodities']
num_assets = len(assets)
expected_returns = np.array([0.085, 0.040, 0.065, 0.045]) 
cov_matrix = np.array([
    [0.0324, 0.0012, 0.0180, 0.0015],
    [0.0012, 0.0064, 0.0016, -0.0004],
    [0.0180, 0.0016, 0.0289, 0.0036],
    [0.0015, -0.0004, 0.0036, 0.0225]
])

# --- MPT MATH ENGINE ---
@st.cache_data
def generate_efficient_frontier(returns, cov_mat, bounds):
    """Sweeps across target returns to map the frontier."""
    target_returns = np.linspace(np.min(returns) + 0.005, np.max(returns) - 0.005, 40)
    frontier_vols, frontier_weights, valid_returns = [], [], []
    
    for target in target_returns:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: np.dot(x, returns) - target}]
        res = sco.minimize(lambda x: np.dot(x.T, np.dot(cov_mat, x)), num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            frontier_vols.append(np.sqrt(res.fun))
            frontier_weights.append(res.x)
            valid_returns.append(target)
            
    return np.array(frontier_vols), np.array(valid_returns), np.array(frontier_weights)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Terminal Controls")
    
    scenario = st.selectbox(
        "Select Tax Research Scenario",
        (
            "Standard Statutory Regime", 
            "Bloomberg Tax Alert: 40% Capital Gains Hike", 
            "Tax Foundation: Bond Protection Act",
            "Custom Interactive Regime"
        )
    )
    
    # --- DYNAMIC CUSTOM SCENARIO SLIDERS ---
    tax_rates = {"equity_tax": 0.20, "bond_tax": 0.35, "real_estate_tax": 0.25, "commodity_tax": 0.28}
    
    if "Bloomberg" in scenario:
        tax_rates["equity_tax"] = 0.40
        tax_rates["real_estate_tax"] = 0.35
    elif "Foundation" in scenario:
        tax_rates["bond_tax"] = 0.15
        tax_rates["equity_tax"] = 0.20
    elif "Custom" in scenario:
        st.markdown("### 🎚️ Custom Tax Adjustments")
        tax_rates["equity_tax"] = st.slider("Equities Tax (%)", 0, 60, 20) / 100.0
        tax_rates["bond_tax"] = st.slider("Fixed Income Tax (%)", 0, 60, 35) / 100.0
        tax_rates["real_estate_tax"] = st.slider("Real Estate Tax (%)", 0, 60, 25) / 100.0
        tax_rates["commodity_tax"] = st.slider("Commodities Tax (%)", 0, 60, 28) / 100.0

    st.markdown("---")
    
    # --- METHODOLOGY SELECTOR ---
    methodology = st.selectbox(
        "Optimization Methodology",
        ("Institutional Bounds (10% - 50%)", "Unconstrained Long-Only (0% - 100%)")
    )
    
    if "Institutional" in methodology:
        active_bounds = tuple((0.10, 0.50) for _ in range(num_assets))
        st.caption("Enforces strict diversification to prevent brittle corner solutions.")
    else:
        active_bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        st.caption("Allows the algorithm to dump 100% of capital into a single asset class if mathematically optimal.")
        
    st.markdown("---")
    
    target_vol = st.slider("Target Portfolio Volatility (Risk %)", min_value=6.0, max_value=14.0, value=10.0, step=0.5) / 100.0


# --- COMPUTE MATH ---
tax_vector = np.array([tax_rates['equity_tax'], tax_rates['bond_tax'], tax_rates['real_estate_tax'], tax_rates['commodity_tax']])
post_tax_returns = expected_returns * (1 - tax_vector)

vols_pre, rets_pre, weights_pre = generate_efficient_frontier(expected_returns, cov_matrix, active_bounds)
vols_post, rets_post, weights_post = generate_efficient_frontier(post_tax_returns, cov_matrix, active_bounds)

# Find the portfolio that best matches the user's selected volatility
if len(vols_pre) > 0 and len(vols_post) > 0:
    idx_pre = (np.abs(vols_pre - target_vol)).argmin()
    idx_post = (np.abs(vols_post - target_vol)).argmin()
else:
    st.error("Algorithm failed to converge. Please adjust bounds or risk parameters.")
    st.stop()

# --- TOP METRICS ---
m1, m2, m3 = st.columns(3)

drag = (np.mean(rets_pre) - np.mean(rets_post)) * 100

m1.markdown(f"<div class='metric-container'><div class='metric-label'>Active Scenario</div><div class='metric-value' style='font-size:18px; margin-top:10px;'>{scenario}</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-container'><div class='metric-label'>Mean Frontier Compression</div><div class='metric-value' style='color:#fc8181;'>-{drag:.2f}%</div><div class='metric-sub'>Yield drag across curve</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-container'><div class='metric-label'>Target Risk Level</div><div class='metric-value' style='color:#68d391;'>{target_vol*100:.1f}% Volatility</div><div class='metric-sub'>Tracking live allocation</div></div>", unsafe_allow_html=True)

st.write("") # Spacer

# --- CHARTS ---
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader("Interactive Efficient Frontier")
    
    fig_line = go.Figure()
    
    # Pre-Tax
    fig_line.add_trace(go.Scatter(
        x=vols_pre*100, y=rets_pre*100, mode='lines', name='Baseline Opportunity Set',
        line=dict(color='#4a5568', width=3)
    ))
    # Post-Tax
    fig_line.add_trace(go.Scatter(
        x=vols_post*100, y=rets_post*100, mode='lines', name='Tax-Compressed Frontier',
        line=dict(color='#63b3ed', width=4)
    ))
    
    # Add Marker for Selected Volatility
    fig_line.add_trace(go.Scatter(
        x=[vols_pre[idx_pre]*100, vols_post[idx_post]*100], 
        y=[rets_pre[idx_pre]*100, rets_post[idx_post]*100], 
        mode='markers', name='Selected Risk Profile',
        marker=dict(color='#f6e05e', size=12, symbol='diamond')
    ))
    
    fig_line.update_layout(
        xaxis_title="Portfolio Volatility / Risk (%)", yaxis_title="Expected Return (%)",
        template="plotly_dark", plot_bgcolor='#0b0f19', paper_bgcolor='#0b0f19',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=30, b=0), height=450
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader(f"Allocation at {target_vol*100:.1f}% Risk")
    
    df_bar = pd.DataFrame({
        'Asset': assets,
        'Pre-Tax Baseline': weights_pre[idx_pre] * 100,
        'Post-Tax Optimized': weights_post[idx_post] * 100
    }).melt(id_vars='Asset', var_name='Regime', value_name='Weight (%)')
    
    fig_bar = px.bar(
        df_bar, x='Weight (%)', y='Asset', color='Regime', barmode='group', orientation='h',
        color_discrete_sequence=['#4a5568', '#63b3ed']
    )
    
    # UI BUG FIX: Moved legend to the top so it doesn't overlap the X-axis label
    fig_bar.update_layout(
        template="plotly_dark", plot_bgcolor='#0b0f19', paper_bgcolor='#0b0f19',
        xaxis_title="Capital Allocation (%)", yaxis_title="",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=30), height=450
    )
    st.plotly_chart(fig_bar, use_container_width=True)
