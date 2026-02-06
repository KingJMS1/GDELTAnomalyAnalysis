from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


Array = jnp.ndarray


@dataclass
class ZINB2FitResult:
    mcmc: MCMC
    posterior_samples: Dict[str, Array]


class ZINB2GLM:
    """
    Bayesian ZINB2 GLM (NB2 variance: Var = mu + alpha * mu^2)

    For a single target series:
      y_i ~ ZINB2(pi_i, mu_i, alpha)
      log mu_i    = H_i^T beta
      logit pi_i  = H_i^T gamma
      alpha       > 0

    Mapping to NumPyro NegativeBinomial2:
      dist.NegativeBinomial2(mean=mu, concentration=k) has Var = mu + mu^2 / k
      so to match Var = mu + alpha * mu^2, we set k = 1/alpha.

    Priors (defaults; you can tune them):
      beta  ~ Normal(0, beta_scale)
      gamma ~ Normal(0, gamma_scale)
      alpha ~ Gamma(alpha_shape, alpha_rate)   # numpyro Gamma uses (concentration, rate)
    """

    def __init__(
        self,
        beta_scale: float = 100.0,
        gamma_scale: float = 100.0,
        alpha_shape: float = 1.0,
        alpha_rate: float = 100.0,
        eps: float = 1e-12,
    ):
        self.beta_scale = float(beta_scale)
        self.gamma_scale = float(gamma_scale)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.eps = float(eps)

        self.fit_result: Optional[ZINB2FitResult] = None

    @staticmethod
    def _nb2_dist(mu: Array, alpha: Array) -> dist.Distribution:
        concentration = 1.0 / alpha
        return dist.NegativeBinomial2(mean=mu, concentration=concentration)

    def _zinb2_loglik(self, y: Array, mu: Array, pi: Array, alpha: Array) -> Array:
        """
        Elementwise ZINB2 log-likelihood for vectors y, mu, pi.

        y:  (n,) nonnegative integers
        mu: (n,) positive
        pi: (n,) in (0,1)
        alpha: scalar > 0
        """
        nb = self._nb2_dist(mu=mu, alpha=alpha)

        log_nb_y = nb.log_prob(y)
        log_nb_0 = nb.log_prob(jnp.zeros_like(y))

        # Clamp pi for numerical stability
        pi = jnp.clip(pi, self.eps, 1.0 - self.eps)

        log_pi = jnp.log(pi)
        log_1m_pi = jnp.log1p(-pi)

        is_zero = (y == 0)

        # y=0: log( pi + (1-pi)*NB(0) )
        log_p0 = logsumexp(
            jnp.stack([log_pi, log_1m_pi + log_nb_0], axis=0),
            axis=0
        )

        # y>0: log(1-pi) + log NB(y)
        log_ppos = log_1m_pi + log_nb_y

        return jnp.where(is_zero, log_p0, log_ppos)

    def model(self, y: Array, H: Array) -> None:
        """
        NumPyro model definition.

        y: (n,) int
        H: (n, d) float
        """
        n, d = H.shape

        beta = numpyro.sample("beta", dist.Normal(0.0, self.beta_scale).expand([d]))
        gamma = numpyro.sample("gamma", dist.Normal(0.0, self.gamma_scale).expand([d]))
        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))

        eta = H @ beta                  # (n,)
        logits_pi = H @ gamma           # (n,)

        mu = jnp.exp(eta)               # (n,)
        pi = jax.nn.sigmoid(logits_pi)  # (n,)

        ll = self._zinb2_loglik(y=y, mu=mu, pi=pi, alpha=alpha)
        numpyro.factor("zinb2_loglik", jnp.sum(ll))

    def fit(
        self,
        y: np.ndarray,
        H: np.ndarray,
        rng_key: "jax.random.PRNGKey",
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
    ) -> ZINB2FitResult:
        """
        Fit via NUTS/HMC.
        """
        y_j = jnp.asarray(y, dtype=jnp.int32)
        H_j = jnp.asarray(H, dtype=jnp.float32)

        kernel = NUTS(
            self.model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(rng_key, y=y_j, H=H_j)

        samples = mcmc.get_samples()
        self.fit_result = ZINB2FitResult(mcmc=mcmc, posterior_samples=samples)
        return self.fit_result

    def _require_fit(self) -> ZINB2FitResult:
        if self.fit_result is None:
            raise RuntimeError("ZINB2GLM is not fit yet. Call fit() first.")
        return self.fit_result

    def link_scale_bands(
        self,
        H_future: np.ndarray,
        q: Tuple[float, float] = (0.025, 0.975),
    ) -> Dict[str, np.ndarray]:
        """
        Credible bands for:
          eta = H beta, mu = exp(eta)
          logits_pi = H gamma, pi = sigmoid(logits_pi)
        """
        fit = self._require_fit()
        s = fit.posterior_samples

        Hf = jnp.asarray(H_future, dtype=jnp.float32)  # (nF, d)

        # shapes:
        # beta: (S, d), gamma: (S, d)
        beta = s["beta"]
        gamma = s["gamma"]

        # (nF, S)
        eta = Hf @ beta.T
        logits_pi = Hf @ gamma.T

        mu = jnp.exp(eta)
        pi = jax.nn.sigmoid(logits_pi)

        def _q(x: Array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            lo = jnp.quantile(x, q[0], axis=1)
            med = jnp.quantile(x, 0.5, axis=1)
            hi = jnp.quantile(x, q[1], axis=1)
            return np.asarray(lo), np.asarray(med), np.asarray(hi)

        eta_lo, eta_med, eta_hi = _q(eta)
        mu_lo, mu_med, mu_hi = _q(mu)
        pi_lo, pi_med, pi_hi = _q(pi)

        return {
            "eta_lo": eta_lo, "eta_med": eta_med, "eta_hi": eta_hi,
            "mu_lo": mu_lo, "mu_med": mu_med, "mu_hi": mu_hi,
            "pi_lo": pi_lo, "pi_med": pi_med, "pi_hi": pi_hi,
        }

    def posterior_predictive_samples(
        self,
        H_future: np.ndarray,
        rng_key: "jax.random.PRNGKey",
        n_draws: Optional[int] = None,
    ) -> Dict[str, Array]:
        """
        Draw posterior predictive samples y_rep at H_future.

        Returns:
          y_rep: (S, nF) int32
          mu:    (S, nF) float
          pi:    (S, nF) float
        """
        fit = self._require_fit()
        s = fit.posterior_samples

        Hf = jnp.asarray(H_future, dtype=jnp.float32)
        nF, d = Hf.shape

        beta = s["beta"]
        gamma = s["gamma"]
        alpha = s["alpha"]  # (S,)

        S_total = beta.shape[0]
        if n_draws is None or n_draws >= S_total:
            idx = jnp.arange(S_total)
        else:
            idx = jnp.linspace(0, S_total - 1, n_draws).astype(jnp.int32)

        beta = beta[idx]
        gamma = gamma[idx]
        alpha = alpha[idx]

        def _one_draw(key, beta_s, gamma_s, alpha_s):
            
            eta = Hf @ beta_s
            logits_pi = Hf @ gamma_s
            mu = jnp.exp(eta)
            pi = jax.nn.sigmoid(logits_pi)

            key_z, key_nb = jax.random.split(key, 2)
            z = dist.Bernoulli(probs=pi).sample(key_z)  # (nF,)
            nb = self._nb2_dist(mu=mu, alpha=alpha_s)
            y_nb = nb.sample(key_nb)                    # (nF,)

            y_rep = jnp.where(z.astype(bool), 0, y_nb)
            return y_rep.astype(jnp.int32), mu, pi

        keys = jax.random.split(rng_key, beta.shape[0])
        y_rep, mu_rep, pi_rep = jax.vmap(_one_draw)(keys, beta, gamma, alpha)

        return {"y_rep": y_rep, "mu": mu_rep, "pi": pi_rep}

    def tail_prob_mc(
        self,
        y_obs: int,
        h_row: np.ndarray,
        rng_key: "jax.random.PRNGKey",
        n_draws: int = 500,
    ) -> float:
        """
        Monte Carlo estimate of right-tail probability:
          P(Y >= y_obs | h_row, data)

        Uses posterior predictive sampling at a single covariate row.
        """
        Hf = np.asarray(h_row, dtype=np.float32).reshape(1, -1)  # (1, d)
        pred = self.posterior_predictive_samples(H_future=Hf, rng_key=rng_key, n_draws=n_draws)
        y_rep = pred["y_rep"]  # (S, 1)
        p = jnp.mean((y_rep[:, 0] >= int(y_obs)).astype(jnp.float32))
        return float(p)
