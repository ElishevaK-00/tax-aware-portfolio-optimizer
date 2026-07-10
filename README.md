# 🏛️ Institutional Tax-Aware Portfolio Optimization

[![Live App](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://tax-aware-portfolio-optimizer-pet4jigvxb9zvswybzsqqs.streamlit.app)

## Research Motivation

**Can natural-language tax policy be translated into mathematically rigorous portfolio optimization constraints?**

Institutional portfolio managers continually adapt allocations in response to changing tax legislation, yet incorporating new regulations into quantitative optimization models remains largely a manual process. This project explores whether modern Large Language Models (LLMs) can bridge that gap by converting tax-policy language into structured mathematical parameters suitable for portfolio optimization.

The result is a research prototype that integrates **Natural Language Processing**, **Convex Optimization**, **Empirical Financial Data**, and **Operations Research** into a single decision-support framework.

---

## Project Overview

The system accepts tax-policy assumptions, converts them into structured optimization parameters, and measures their impact on institutional portfolio construction using the Markowitz Mean-Variance framework.

Using **10 years of empirical market data**, the optimization engine recalculates the efficient frontier under changing tax regimes while preserving institutional allocation constraints.

The platform is intended as a proof-of-concept investigating the interaction between

- Tax Policy
- Portfolio Optimization
- Mathematical Programming
- Explainable AI
- Financial Decision Support Systems

rather than as an investment recommendation system.

---

## Methodology

### 1. Policy Interpretation

Natural-language tax scenarios are processed through an LLM-inspired parsing pipeline that extracts structured tax parameters across multiple asset classes.

Example:

```
Increase capital gains tax to 40%.
Reduce fixed-income taxation to 15%.
```

↓

```
Equities = 40%
Fixed Income = 15%
Real Estate = 25%
Commodities = 28%
```

Every extracted parameter is recorded in an auditable computation log.

---

### 2. Market Data

Historical prices are downloaded using **Yahoo Finance** (`yfinance`).

The system computes

- Daily logarithmic returns
- Annualized expected returns
- Annualized covariance matrix

from approximately ten years of historical observations.

Assets currently include

- SPY (Equities)
- AGG (Fixed Income)
- VNQ (Real Estate)
- GLD (Commodities)

---

### 3. Convex Portfolio Optimization

Tax-adjusted expected returns are computed as

\[
\mu_{post} = \mu_{pre} \odot (1-T)
\]

where

- **μ** represents expected returns
- **T** represents the tax vector extracted from policy

The optimization problem is formulated as

\[
\begin{aligned}
\min_w \quad & w^T\Sigma w \\
\text{s.t.}\quad
& \mu^T w \ge R_{target} \\
& \sum_i w_i = 1 \\
& l_i \le w_i \le u_i
\end{aligned}
\]

using **CVXPY** and quadratic programming.

Institutional allocation limits (10–50%) are enforced mathematically during optimization.

---

### 4. Comparative Analysis

The application dynamically compares

- Pre-tax efficient frontier
- Post-tax efficient frontier
- Frontier compression
- Allocation shifts
- Estimated regime-transition costs

allowing users to evaluate how hypothetical tax regimes influence optimal portfolio construction.

---

## Features

- Live market data integration
- Mean-Variance Optimization (Markowitz)
- Convex Optimization using CVXPY
- Dynamic efficient frontier generation
- Interactive institutional allocation analysis
- Tax-aware portfolio optimization
- Transparent NLP audit trail
- Interactive Streamlit dashboard

---

## Technology Stack

### Optimization

- CVXPY
- OSQP
- Clarabel

### Data

- NumPy
- Pandas
- yfinance

### Artificial Intelligence

- OpenAI API
- Natural Language Processing
- Structured parameter extraction

### Visualization

- Plotly
- Streamlit

---

## Repository Structure

```
├── app.py
├── requirements.txt
├── assets/
├── screenshots/
└── README.md
```

---

## Current Limitations

This project is intended as a research prototype.

Current assumptions include

- deterministic tax rates
- static expected-return estimation
- simplified tax treatment by asset class
- hypothetical policy parsing

Future work includes

- stochastic tax-policy simulation
- multi-period optimization
- transaction-cost calibration
- direct parsing of IRS bulletins and Treasury releases
- reinforcement learning for dynamic tax-aware allocation
- institutional stress-testing under evolving regulatory regimes

---

## Disclaimer

This project is intended solely for educational and research purposes.

It does not constitute financial, investment, tax, or legal advice.
