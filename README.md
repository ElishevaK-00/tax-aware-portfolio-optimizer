# 🏛️ LLM-Driven Tax-Aware Portfolio Optimizer

[![Live App](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://tax-aware-portfolio-optimizer-pet4jigvxb9zvswybzsqqs.streamlit.app)

**A proof-of-concept translating unstructured tax policy into mathematical portfolio constraints.**

This repository demonstrates an experimental architecture bridging **Natural Language Processing (LLMs)** and **Operations Research (Mean-Variance Optimization)**. By securely parsing hypothetical tax regulations via the OpenAI API into structured JSON parameters, the engine mathematically penalizes expected returns to optimize for the maximum *after-tax* Sharpe Ratio.

### ⚙️ Methodology Pipeline
1. **Unstructured Input:** Accepts raw text outlining hypothetical or proposed tax regulations.
2. **LLM Parameter Extraction:** Connects to OpenAI (`gpt-4o` in JSON mode) to extract specific tax rates for Equities, Bonds, Real Estate, and Commodities.
3. **Pre-Tax vs. Post-Tax Optimization:** Feeds the adjusted expected returns into a SciPy optimizer (`SLSQP`) to minimize portfolio variance while maximizing risk-adjusted yield.
4. **Interactive Visualization:** Renders dynamic weight-shifting using Plotly to visually demonstrate capital reallocation under the new tax regime.

### 🚀 Tech Stack
* **Frontend/Deployment:** Streamlit
* **Optimization Engine:** SciPy, NumPy, Pandas
* **Visualization:** Plotly Express
* **NLP Parsing (Architecture):** OpenAI Python SDK
