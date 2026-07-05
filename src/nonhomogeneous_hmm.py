import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import warnings

class NonHomogeneousGaussianHMM:
    def __init__(self, n_components=3, n_iter=100, tol=1e-4, n_restarts=3, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state
        
        self.A_event_ = None
        self.A_normal_ = None
        self.pi_ = None
        self.means_ = None
        self.covars_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def _init_parameters(self, X, rng):
        N = self.n_components
        n_features = X.shape[1]
        
        # Init pi
        self.pi_ = np.ones(N) / N
        
        # Init means via KMeans + noise
        kmeans = KMeans(n_clusters=N, random_state=rng.randint(10000), n_init=10)
        kmeans.fit(X)
        self.means_ = kmeans.cluster_centers_ + rng.normal(0, 0.01, size=(N, n_features))
        
        # Init covars (diagonal covariances)
        # We store them as N x n_features array of variances
        self.covars_ = np.var(X, axis=0) * np.ones((N, n_features))
        
        # Init A_event and A_normal using Dirichlet(alpha=5*I + 1)
        alpha = 5 * np.eye(N) + 1
        self.A_event_ = np.zeros((N, N))
        self.A_normal_ = np.zeros((N, N))
        for i in range(N):
            self.A_event_[i] = rng.dirichlet(alpha[i])
            self.A_normal_[i] = rng.dirichlet(alpha[i])

    def _compute_log_b(self, X):
        T, n_features = X.shape
        N = self.n_components
        log_b = np.zeros((T, N))
        
        for i in range(N):
            # We use multivariate normal with diagonal covariance
            cov = np.diag(self.covars_[i] + 1e-6)
            log_b[:, i] = multivariate_normal.logpdf(X, mean=self.means_[i], cov=cov)
        return log_b

    def _log_forward(self, log_b, E, A_event=None, A_normal=None):
        if A_event is None: A_event = self.A_event_
        if A_normal is None: A_normal = self.A_normal_
        
        T, N = log_b.shape
        log_alpha = np.full((T, N), -np.inf)
        
        # Initialization
        log_alpha[0] = np.log(self.pi_ + 1e-300) + log_b[0]
        
        # Vectorized Recursion
        for t in range(1, T):
            A_t = A_event if E[t] else A_normal
            # log_alpha[t, j] = logsumexp(log_alpha[t-1] + log(A_t[:, j])) + log_b[t, j]
            # We do it efficiently:
            # A_t is (N, N). We want to sum over i.
            # log_alpha[t-1] is shape (N,). log(A_t) is shape (N, N)
            
            work_buffer = log_alpha[t-1][:, np.newaxis] + np.log(A_t + 1e-300)
            log_alpha[t] = logsumexp(work_buffer, axis=0) + log_b[t]
            
            # Explicit loop version as reference:
            # for j in range(N):
            #     log_alpha[t, j] = logsumexp(log_alpha[t-1] + np.log(A_t[:, j] + 1e-300))
            #     log_alpha[t, j] += log_b[t, j]
                
        return log_alpha

    def _log_backward(self, log_b, E, A_event=None, A_normal=None):
        if A_event is None: A_event = self.A_event_
        if A_normal is None: A_normal = self.A_normal_
        
        T, N = log_b.shape
        log_beta = np.full((T, N), -np.inf)
        
        # Initialization
        log_beta[T-1] = 0.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            A_t1 = A_event if E[t+1] else A_normal
            
            work_buffer = np.log(A_t1 + 1e-300) + log_b[t+1] + log_beta[t+1]
            log_beta[t] = logsumexp(work_buffer, axis=1)
            
        return log_beta

    def _do_em_step(self, X, E, log_b, log_alpha, log_beta, log_likelihood):
        T, N = log_b.shape
        n_features = X.shape[1]
        
        # Compute gamma
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        
        # Compute xi
        # We need xi split into event and normal buckets
        # xi_event[i, j] = sum over t (where E[t+1]=1) of xi_t(i, j)
        xi_event_sum = np.zeros((N, N))
        xi_normal_sum = np.zeros((N, N))
        
        gamma_event_sum = np.zeros(N)
        gamma_normal_sum = np.zeros(N)
        
        for t in range(T-1):
            A_t1 = self.A_event_ if E[t+1] else self.A_normal_
            
            log_xi_t = (log_alpha[t][:, np.newaxis] + 
                        np.log(A_t1 + 1e-300) + 
                        log_b[t+1] + 
                        log_beta[t+1] - 
                        log_likelihood)
            xi_t = np.exp(log_xi_t)
            
            if E[t+1]:
                xi_event_sum += xi_t
                gamma_event_sum += gamma[t]
            else:
                xi_normal_sum += xi_t
                gamma_normal_sum += gamma[t]
                
        # M-step Updates
        
        # 1. Update pi
        self.pi_ = gamma[0] / np.sum(gamma[0])
        
        # 2. Update Transitions
        for i in range(N):
            if gamma_event_sum[i] > 1e-6:
                self.A_event_[i] = xi_event_sum[i] / gamma_event_sum[i]
                # Normalize just in case
                self.A_event_[i] /= np.sum(self.A_event_[i])
            
            if gamma_normal_sum[i] > 1e-6:
                self.A_normal_[i] = xi_normal_sum[i] / gamma_normal_sum[i]
                self.A_normal_[i] /= np.sum(self.A_normal_[i])
                
        # 3. Update Emissions (Pooled)
        gamma_total = np.sum(gamma, axis=0)
        for i in range(N):
            if gamma_total[i] > 1e-6:
                # Means
                self.means_[i] = np.sum(gamma[:, i:i+1] * X, axis=0) / gamma_total[i]
                
                # Covars (diagonal)
                diff = X - self.means_[i]
                self.covars_[i] = np.sum(gamma[:, i:i+1] * (diff ** 2), axis=0) / gamma_total[i]
                self.covars_[i] += 1e-6 # Add jitter for stability

    def _fit_single_restart(self, X, E, rng):
        self._init_parameters(X, rng)
        
        old_ll = -np.inf
        
        for iteration in range(self.n_iter):
            log_b = self._compute_log_b(X)
            log_alpha = self._log_forward(log_b, E)
            
            ll = logsumexp(log_alpha[-1])
            
            if np.abs(ll - old_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
                
            old_ll = ll
            log_beta = self._log_backward(log_b, E)
            self._do_em_step(X, E, log_b, log_alpha, log_beta, ll)
            
        return ll

    def fit(self, X, E):
        best_ll = -np.inf
        best_params = None
        rng = np.random.RandomState(self.random_state)
        
        for restart in range(self.n_restarts):
            # Temporary state variables are mutated directly on self
            ll = self._fit_single_restart(X, E, rng)
            
            if ll > best_ll:
                best_ll = ll
                best_params = {
                    'A_event_': self.A_event_.copy(),
                    'A_normal_': self.A_normal_.copy(),
                    'pi_': self.pi_.copy(),
                    'means_': self.means_.copy(),
                    'covars_': self.covars_.copy(),
                    'converged_': self.converged_,
                    'n_iter_': self.n_iter_
                }
                
        # Restore best params
        if best_params is not None:
            for k, v in best_params.items():
                setattr(self, k, v)
                
        return self

    def score(self, X, E, A_event_override=None):
        log_b = self._compute_log_b(X)
        log_alpha = self._log_forward(log_b, E, A_event=A_event_override)
        return logsumexp(log_alpha[-1])

    def decode(self, X, E):
        log_b = self._compute_log_b(X)
        T, N = log_b.shape
        
        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        
        log_delta[0] = np.log(self.pi_ + 1e-300) + log_b[0]
        
        for t in range(1, T):
            A_t = self.A_event_ if E[t] else self.A_normal_
            
            work_buffer = log_delta[t-1][:, np.newaxis] + np.log(A_t + 1e-300)
            
            log_delta[t] = np.max(work_buffer, axis=0) + log_b[t]
            psi[t] = np.argmax(work_buffer, axis=0)
            
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
            
        return states
