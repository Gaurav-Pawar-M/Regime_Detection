# Derivation of the Non-Homogeneous Hidden Markov Model (NH-HMM)

## 1. Standard HMM (Time-Homogeneous)

We define the standard HMM with the following notation:
- $T$: Total number of trading days.
- $N$: Number of hidden regimes (states).
- $S_t \in \{1, \dots, N\}$: Hidden state at time $t$.
- $O_t$: Observation vector at time $t$ (e.g., log returns and volatility).
- $A$: Transition matrix, where $A[i,j] = P(S_t = j \mid S_{t-1} = i)$.
- $\pi_i$: Initial state probability, $P(S_1 = i)$.
- $b_i(O_t)$: Emission probability/density, $P(O_t \mid S_t = i)$. For a Gaussian HMM, this is parameterized by mean $\mu_i$ and variance $\sigma_i^2$.

### The Forward Algorithm
The forward variable is defined as $\alpha_t(i) = P(O_{1:t}, S_t = i \mid \theta)$.
**Initialization:**
$$ \alpha_1(i) = \pi_i b_i(O_1) $$
**Recursion:**
$$ \alpha_t(j) = \left( \sum_{i=1}^N \alpha_{t-1}(i) A[i,j] \right) b_j(O_t) $$

### The Backward Algorithm
The backward variable is defined as $\beta_t(i) = P(O_{t+1:T} \mid S_t = i, \theta)$.
**Initialization:**
$$ \beta_T(i) = 1 $$
**Recursion:**
$$ \beta_t(i) = \sum_{j=1}^N A[i,j] b_j(O_{t+1}) \beta_{t+1}(j) $$

### The Viterbi Algorithm
Finds the single most likely path of hidden states.
**Initialization:**
$$ \delta_1(i) = \pi_i b_i(O_1) $$
**Recursion:**
$$ \delta_t(j) = \max_i \left( \delta_{t-1}(i) A[i,j] \right) b_j(O_t) $$

### Baum-Welch M-Step (Standard)
Using $\gamma_t(i) = P(S_t = i \mid O, \theta)$ and $\xi_t(i,j) = P(S_t = i, S_{t+1} = j \mid O, \theta)$, the transition matrix is updated as:
$$ A[i,j]^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)} $$

---

## 2. The Non-Homogeneous Extension: Motivation

The standard homogeneous assumption treats every trading day as informationally equivalent. In reality, the NSE event calendar provides a known, observed, forward-looking indicator of days with elevated regime-switching probability (such as earnings announcements, board meetings, or AGMs). Incorporating it requires nothing more than allowing the transition matrix $A$ to be indexed by time $t$, dependent on event proximity.

---

## 3. Definition of $A_t$

We introduce a binary event indicator $E_t$:
$$ E_t = 1 \quad \text{if } |\text{days\_to\_nearest\_event}_t| \le 5 \text{ trading days, else } 0 $$

The transition matrix governing the move **into** day $t$ is chosen as:
$$
A_t = 
\begin{cases} 
A_{\text{event}} & \text{if } E_t = 1 \\
A_{\text{normal}} & \text{if } E_t = 0
\end{cases}
$$
*Explicit Convention:* $A_t$ governs the transition from day $t-1$ to day $t$.

---

## 4. Modified Algorithms

Because the transition probabilities now depend on $t$, we substitute $A$ with $A_t$ (or $A_{t+1}$) exactly where the transition occurs.

### Forward Algorithm, Modified
**Recursion:**
$$ \alpha_t(j) = \left( \sum_{i=1}^N \alpha_{t-1}(i) A_t[i,j] \right) b_j(O_t) $$

### Backward Algorithm, Modified
**Recursion:**
$$ \beta_t(i) = \sum_{j=1}^N A_{t+1}[i,j] b_j(O_{t+1}) \beta_{t+1}(j) $$
*(Notice we use $A_{t+1}$ because it represents the transition from $t$ to $t+1$.)*

### Viterbi Algorithm, Modified
**Recursion:**
$$ \delta_t(j) = \max_i \left( \delta_{t-1}(i) A_t[i,j] \right) b_j(O_t) $$

### Baum-Welch M-Step, Modified
The emission parameters $\mu_i, \sigma_i^2$ are **not split**, because the regime's return personality is assumed constant; only the probability of entering/leaving it varies with event proximity. However, the transition update is split into two buckets based on $E_{t+1}$:

$$ A_{\text{event}}[i,j]^{\text{new}} = \frac{\sum_{t: E_{t+1}=1} \xi_t(i,j)}{\sum_{t: E_{t+1}=1} \gamma_t(i)} $$

$$ A_{\text{normal}}[i,j]^{\text{new}} = \frac{\sum_{t: E_{t+1}=0} \xi_t(i,j)}{\sum_{t: E_{t+1}=0} \gamma_t(i)} $$

---

## 5. Mathematical Validity

The Markov property is preserved conditional on the observed covariate $E_t$. At every step, the next state's distribution is fully determined by the current state and $A_t$. Because $A_t$ is determined by an observed, non-hidden covariate (the corporate event calendar), this constitutes a valid Time-Varying Transition Probability (TVTP) Markov switching model. Canonical references for this class include Hamilton (1989) and Filardo (1994).

---

## 6. Limitations of hmmlearn

The `hmmlearn` library's `GaussianHMM` operates using a single `transmat_` attribute. Every internal forward, backward, and Viterbi call indexes this single matrix, and there is no API to pass a per-timestep transition matrix sequence. Furthermore, the `_do_mstep()` function unconditionally pools all $\xi_t(i,j)$ across time. Both of these hardcoded assumptions must be overridden, necessitating a from-scratch numpy implementation.

---

## 7. Correctness Check

If we artificially force $A_{\text{event}} := A_{\text{normal}}$, the NH-HMM must return the exact same log-likelihood and decoded states as the standard homogeneous HMM on identical data. This is a formal mathematical reduction: setting the matrices equal removes the covariate dependence, algebraically collapsing the non-homogeneous model back to the standard homogeneous model.
