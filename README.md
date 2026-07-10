# 🏛️ Institutional Tax-Aware Portfolio Terminal

[![Live App](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://tax-aware-portfolio-optimizer-pet4jigvxb9zvswybzsqqs.streamlit.app)

**A quantitative proof-of-concept translating unstructured macroeconomic tax policy into hard mathematical portfolio constraints.**

This repository demonstrates an experimental architecture bridging **Natural Language Processing (LLMs)**, **Accounting Auditability**, and **Operations Research**. By securely parsing hypothetical tax regulations into structured parameters, the engine applies convex optimization to mathematically map the structural compression of the *after-tax* Markowitz Efficient Frontier.

### ⚙️ Methodology Pipeline
1. **Unstructured Input:** Accepts raw text outlining hypothetical or proposed tax regulations.
2. **NLP Parameter Vectorization:** Simulates an LLM parsing the text to extract specific tax rates across asset classes, logging the exact translation in a transparent audit ledger.
3. **Convex Optimization (CVXPY):** Feeds the adjusted expected returns into a rigorous CVXPY quadratic programming engine. The solver minimizes portfolio variance while enforcing strict institutional diversification bounds (10% - 50%).
4. **Interactive Visualization:** Dynamically charts the baseline vs. tax-compressed opportunity sets, calculating mean frontier drag and regime transition friction.

### 🚀 Tech Stack
* **Frontend/Deployment:** Streamlit
* **Convex Optimization Engine:** CVXPY (Quadratic Programming / ECOS Solvers)
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Plotly Express
