import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
from openai import OpenAI
import json
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Institutional Tax-Aware Optimizer", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS FOR TERMINAL UI ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #a0aec0; }
    h1, h2, h3 { color: #63b3ed; font-family: 'Inter', sans-serif; }
    .stChatInputContainer { padding-bottom: 20px; }
    .metric-box { background-color: #1a202c; padding: 15px; border-radius: 8px; border-left: 4px solid #63b3ed; margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Institutional Tax-Aware Portfolio Terminal")
st.markdown("Automated NLP parameter extraction & Efficient Frontier mapping.")
st.divider()

# --- INITIALIZE SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System Initialized. I am your Quantitative Tax Assistant. Please describe a hypothetical macro tax policy, and I will extract the asset-class implications and map the structural shift on the Efficient Frontier."}
    ]
if "current_tax_rates" not in st.session_state:
    st.session_state.current_tax_rates = {"equity_tax": 0.20, "bond_tax": 0.35, "real_estate_tax": 0.25, "commodity_tax": 0.28}

# --- EMPIRICALLY ANCHORED CMAs ---
assets = ['Global Equities', 'Fixed Income', 'Real Estate', 'Commodities']
num_assets = len(assets)
expected_returns = np.array([0.085, 0.040, 0.065, 0.045]) 
cov_matrix = np.array([
    [0.0324, 0.0012, 0.0180, 0.0015],
    [0.0012, 0.0064, 0.0016, -0.0004],
    [0.0180, 0.0016, 0.0289, 0.0036],
    [0.0015, -0.0004, 0.0036, 0.0225]
])

# --- AI / MOCK PARSER ENGINE ---
def extract_tax_constraints(prompt):
    """
    Attempts to use OpenAI API from Streamlit Secrets.
    Falls back to a smart regex/mock engine if no key is provided, preventing crashes.
    """
    try:
        # Attempt to pull key from Streamlit's secure backend
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
        system_prompt = '''You are a quantitative tax engine. Extract tax rates for Equities, Bonds, Real Estate, and Commodities from the text. Return ONLY JSON with keys: "equity_tax", "bond_tax", "real_estate_tax", "commodity_tax". Use float values (e.g., 0.39). Default standard rates if unmentioned: Eq:0.20, Bd:0.35, RE:0.25, Com:0.28.'''
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content), "Live API"
    except Exception:
        # Fallback Logic (Demo Mode) if API key fails or isn't set
        time.sleep(1) # Simulate thinking
        rates = {"equity_tax": 0.20, "bond_tax": 0.35, "real_estate_tax": 0.25, "commodity_tax": 0.28}
        prompt_lower = prompt.lower()
        if "40" in prompt_lower or "increase" in prompt_lower:
            rates["equity_tax"] = 0.40
            rates["real_estate_tax"] = 0.35
        if "cut" in prompt_lower or "15" in prompt_lower:
            rates["equity_tax"] = 0.15
            rates["bond_tax"] = 0.20
        return rates, "Local Simulation"

# --- MPT MATH ENGINE ---
def generate_efficient_frontier(returns, cov_mat, target_returns):
    frontier_vols, frontier_weights = [], []
    bounds = tuple((0.10, 0.50) for _ in range(num_assets)) # Institutional limits
    for target in target_returns:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: np.dot(x, returns) - target}]
        res = sco.minimize(lambda x: np.dot(x.T, np.dot(cov_mat, x)), num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            frontier_vols.append(np.sqrt(res.fun))
            frontier_weights.append(res.x)
        else:
            frontier_vols.append(None)
            frontier_weights.append(None)
    return frontier_vols, frontier_weights

# --- DASHBOARD LAYOUT ---
col_chat, col_viz = st.columns([1.2, 2], gap="large")

with col_chat:
    st.subheader("💬 Tax Policy Assistant")
    
    # Render chat history
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat Input
    if prompt := st.chat_input("Enter a proposed tax regulation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Process AI response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing policy parameters..."):
                    new_rates, mode = extract_tax_constraints(prompt)
                    st.session_state.current_tax_rates = new_rates
                    
                    reply = f"**[{mode} Mode]** Analysis complete. Extracted parametric constraints:\n\n"
                    reply += f"- **Equities:** {new_rates['equity_tax']*100}%\n"
                    reply += f"- **Bonds:** {new_rates['bond_tax']*100}%\n"
                    reply += f"- **Real Estate:** {new_rates['real_estate_tax']*100}%\n"
                    reply += f"- **Commodities:** {new_rates['commodity_tax']*100}%\n\n"
                    reply += "Updating institutional frontier models now."
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

with col_viz:
    st.subheader("📈 Interactive Markowitz Frontier")
    
    # Calculate Data
    tax_vector = np.array(list(st.session_state.current_tax_rates.values()))
    post_tax_returns = expected_returns * (1 - tax_vector)
    
    target_returns_pre = np.linspace(0.045, 0.065, 30)
    target_returns_post = np.linspace(0.025, 0.048, 30)
    
    vols_pre, weights_pre = generate_efficient_frontier(expected_returns, cov_matrix, target_returns_pre)
    vols_post, weights_post = generate_efficient_frontier(post_tax_returns, cov_matrix, target_returns_post)
    
    valid_pre = [(v, r, w) for v, r, w in zip(vols_pre, target_returns_pre, weights_pre) if v is not None]
    valid_post = [(v, r, w) for v, r, w in zip(vols_post, target_returns_post, weights_post) if v is not None]
    
    # Plotly Graph
    fig = go.Figure()
    
    # Pre-Tax Line
    fig.add_trace(go.Scatter(
        x=[x[0]*100 for x in valid_pre], y=[x[1]*100 for x in valid_pre],
        mode='lines', name='Baseline Opportunity Set',
        line=dict(color='#4a5568', width=3),
        text=[f"Equities: {x[2][0]*100:.1f}%<br>Bonds: {x[2][1]*100:.1f}%<br>RE: {x[2][2]*100:.1f}%<br>Cmdty: {x[2][3]*100:.1f}%" for x in valid_pre],
        hoverinfo='text+x+y'
    ))
    
    # Post-Tax Line
    fig.add_trace(go.Scatter(
        x=[x[0]*100 for x in valid_post], y=[x[1]*100 for x in valid_post],
        mode='lines+markers', name='Post-Tax Structural Frontier',
        line=dict(color='#63b3ed', width=3),
        marker=dict(size=6, color='#63b3ed'),
        text=[f"Equities: {x[2][0]*100:.1f}%<br>Bonds: {x[2][1]*100:.1f}%<br>RE: {x[2][2]*100:.1f}%<br>Cmdty: {x[2][3]*100:.1f}%" for x in valid_post],
        hoverinfo='text+x+y'
    ))
    
    fig.update_layout(
        xaxis_title="Portfolio Volatility (Risk %)", yaxis_title="Expected Portfolio Return (%)",
        template="plotly_dark", plot_bgcolor='#0b0f19', paper_bgcolor='#0b0f19',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=10, r=10, t=10, b=10), height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Live KPI Tiles below the graph
    st.markdown("### Structural Diagnostics")
    m1, m2, m3 = st.columns(3)
    
    implied_drag = (np.mean(target_returns_pre) - np.mean(target_returns_post)) * 100
    
    m1.markdown(f"""<div class='metric-box'><div style='color:#a0aec0; font-size:14px;'>Active Tax Regime</div><div style='font-size:24px; color:white; font-weight:bold;'>Custom Input</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class='metric-box'><div style='color:#a0aec0; font-size:14px;'>Mean Frontier Drag</div><div style='font-size:24px; color:#fc8181; font-weight:bold;'>-{implied_drag:.2f}%</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class='metric-box'><div style='color:#a0aec0; font-size:14px;'>Algorithm Status</div><div style='font-size:24px; color:#68d391; font-weight:bold;'>SLSQP Constrained</div></div>""", unsafe_allow_html=True)
