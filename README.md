# NSE Event-Conditioned Regime-Switching Engine

This repository contains the complete implementation for the quantitative research project exploring a **Non-Homogeneous Hidden Markov Model (NH-HMM)** tailored to the Indian equity market (NSE). 

## Motivating Gap
Standard HMMs assume a homogeneous (constant) transition matrix over time. In financial markets, however, regime shifts (e.g. from Bull to Bear) are significantly more likely to occur immediately following scheduled information disclosure events such as quarterly earnings or board meetings. 

This project bridges that gap by implementing a TVTP (Time-Varying Transition Probability) HMM where the transition matrix switches between $A_{\text{normal}}$ and $A_{\text{event}}$ depending on whether the trading day falls within a $\pm5$ day window of a scheduled NSE corporate event.

## Key Findings
- **Precision:** Manual hand-verification of the disagreement cases between the Baseline HMM and the new NH-HMM showed high precision in correctly identifying fundamentally justified regime shifts near corporate events.
- **Quantitative Validation:** The NH-HMM's precision was rigorously evaluated via a long/avoid backtest strategy. Layering this regime signal with technical indicators via an XGBoost classifier produced significant predictive advantages as proven by SHAP value analysis.

## Project Structure
- `data/`: Parquet files comprising the master trading calendar, raw returns, scraped NSE events, and generated regime states.
- `notebooks/`: Sequential execution flow from data acquisition (01) through Baseline HMM (02), Event scraping (03), NH-HMM fit (05), Per-stock scaling (06), Checkpointing (07), Quantitative Validation (10), and Feature Attribution (11).
- `src/`: Core logic including the custom from-scratch `NonHomogeneousGaussianHMM` implementation utilizing `scipy.special.logsumexp` for numerical stability, and the `LayeredRegimeClassifier`.
- `reports/`: The mathematical derivation for the modified Baum-Welch steps (`derivation_nonhomogeneous_hmm.md`) and the final project write-up (`final_writeup.md`).
- `app/`: The interactive Streamlit dashboard allowing users to visualize regime histories overlaid with SHAP-explained AI buy/hold/avoid signals.

## Running the Dashboard
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
