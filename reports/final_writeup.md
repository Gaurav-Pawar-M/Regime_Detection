# Final Write-up: NSE Event-Conditioned Regime-Switching Engine

## 1. Project Overview
This project successfully implemented a **Non-Homogeneous Hidden Markov Model (NH-HMM)** tailored to the Indian Equity Market. Standard HMMs use a static transition matrix $A$. However, financial markets exhibit structural regime shifts directly following the public disclosure of material information (e.g., quarterly results, board meetings). We replaced the static matrix with a Time-Varying Transition Probability (TVTP) matrix that switches between $A_{\text{normal}}$ and $A_{\text{event}}$ depending on a $\pm5$ day window around scheduled NSE corporate events.

## 2. Mathematical Derivation
The derivation of the Event-Conditioned HMM required modifying the core Expectation-Maximization (Baum-Welch) algorithm. The traditional Forward ($\alpha_t$) and Backward ($\beta_t$) variables were updated to index the time-dependent transition matrix $A_t$:

$$ \alpha_t(j) = P(O_{1:t}, S_t = j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) A_t(i, j) \right] b_j(O_t) $$

where $A_t = A_{\text{event}}$ if an event is scheduled within the window, and $A_{\text{normal}}$ otherwise. The M-Step was bifurcated to compute $\xi_t$ parameters conditionally. By aggregating $\xi_t$ arrays explicitly partitioned by $E_t \in \{0,1\}$, we solved for the dual matrices simultaneously. All calculations were implemented natively in Python using `scipy.special.logsumexp` to guarantee numerical stability across 5 years of log-likelihood additions.

## 3. Hand-Verification Methodology
To validate the model without succumbing to data snooping, a strict manual verification process was followed. 

1. **Disagreement Mining:** We ran both the standard Baseline HMM and our newly built NH-HMM across 18 distinct NSE equities over 5 years.
2. **Filtration:** We computationally identified the exact subset of trading days where the two models assigned conflicting regime states *specifically* on the day of a scheduled corporate event.
3. **Manual Validation:** By reading the actual NSE PDF filings associated with those specific event dates, we verified whether the new regime state called by the NH-HMM was fundamentally justified by the disclosed numbers (e.g., a massive earnings beat justifying a Bull shift, or a severe guidance cut justifying a Bear shift). 

**Result:** The verification set demonstrated exceptional precision for the NH-HMM. The model frequently captured rapid sentiment shifts that the Baseline HMM, smoothed by its homogeneous transition prior, entirely missed.

## 4. Quantitative Results & Layered AI
We backtested a standard long/avoid strategy gated on the predicted regimes. The NH-HMM demonstrated stronger performance within event windows, capturing the upside of positive disclosures faster. 

To operationalize the signal, an `XGBoostClassifier` was layered over the model, accepting features such as `nh_state`, `momentum_20d`, `vol_ratio`, and `days_to_event`. 
Using **SHAP (SHapley Additive exPlanations)**, we formally proved that the `days_to_event` feature and the `nh_state` significantly governed the directional logic of the final buy/hold/avoid prediction.

## 5. Concrete Examples
- **Successful Identification (DRREDDY):** On the day of a Q4 results announcement in 2019, the pharmaceutical sector faced severe margin pressures. The Baseline HMM, smoothed by the previous month of sideways trading, failed to immediately downgrade the regime. The NH-HMM instantly shifted into the Bear regime due to the elevated $A_{\text{event}}$ transition probability, accurately protecting capital from the subsequent week's drawdown.
- **Ambiguous Read:** During a generic AGM in a calm market, the NH-HMM occasionally triggered a rapid state transition due to the sheer presence of $E_t=1$, even when no material new guidance was provided by the board. This highlighted the need for the layered XGBoost approach to filter out "empty" scheduled events using technical momentum context.

## 6. Conclusion
The NH-HMM proves that integrating deterministic corporate calendars explicitly into the transition priors of unobserved state models provides a significant edge in equity market analysis. By bridging mathematical rigor with manual fundamental verification and layered Machine Learning deployment, this project serves as a comprehensive end-to-end quantitative pipeline.
