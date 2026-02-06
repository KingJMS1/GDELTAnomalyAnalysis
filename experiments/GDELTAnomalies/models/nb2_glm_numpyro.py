from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


Array = jnp.ndarray


@dataclass
class NB2FitResult:
    mcmc: MCMC
    posterior_samples: Dict[str, Array]


class NB2GLM:
    """
    Bayesian NB2 GLM (Var = mu + alpha * mu^2)

    Single target series:
      y_i ~ NB2(mu_i, alpha)
      log mu_i = H_i^T beta
      alpha > 0

    Mapping to NumPyro NegativeBinomial2:
      Var = mu + mu^2 / concentration
      => concentration = 1/alpha
    """

    def __init__(
        self,
        beta_scale: float = 10.0,
        alpha_shape: float = 1.0,
        alpha_rate: float = 100.0,
    ):
        self.beta_scale = float(beta_scale)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.fit_result: Optional[NB2FitResult] = None

    @staticmethod
    def _nb2_dist(mu: Array, alpha: Array) -> dist.Distribution:
        concentration = 1.0 / alpha
        return dist.NegativeBinomial2(mean=mu, concentration=concentration)

    def model(self, y: Array, H: Array) -> None:
        """
        y: (n,) int
        H: (n, d) float
        """
        n, d = H.shape

        beta = numpyro.sample("beta", dist.Normal(0.0, self.beta_scale).expand([d]))
        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))

        eta = H @ beta
        mu = jnp.exp(eta)

        nb = self._nb2_dist(mu=mu, alpha=alpha)
        numpyro.sample("y", nb, obs=y)

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
    ) -> NB2FitResult:
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
        self.fit_result = NB2FitResult(mcmc=mcmc, posterior_samples=samples)
        return self.fit_result

    def _require_fit(self) -> NB2FitResult:
        if self.fit_result is None:
            raise RuntimeError("NB2GLM is not fit yet. Call fit() first.")
        return self.fit_result

    def link_scale_bands(
        self,
        H_future: np.ndarray,
        q: Tuple[float, float] = (0.025, 0.975),
    ) -> Dict[str, np.ndarray]:
        """
        Credible bands for:
          eta = H beta, mu = exp(eta)
        """
        fit = self._require_fit()
        s = fit.posterior_samples

        Hf = jnp.asarray(H_future, dtype=jnp.float32)
        beta = s["beta"]  # (S, d)

        eta = Hf @ beta.T     # (nF, S)
        mu = jnp.exp(eta)

        def _q(x: Array):
            lo = jnp.quantile(x, q[0], axis=1)
            med = jnp.quantile(x, 0.5, axis=1)
            hi = jnp.quantile(x, q[1], axis=1)
            return np.asarray(lo), np.asarray(med), np.asarray(hi)

        eta_lo, eta_med, eta_hi = _q(eta)
        mu_lo, mu_med, mu_hi = _q(mu)

        return {
            "eta_lo": eta_lo, "eta_med": eta_med, "eta_hi": eta_hi,
            "mu_lo": mu_lo, "mu_med": mu_med, "mu_hi": mu_hi,
        }

    def posterior_predictive_samples(
        self,
        H_future: np.ndarray,
        rng_key: "jax.random.PRNGKey",
        n_draws: Optional[int] = None,
    ) -> Dict[str, Array]:
        """
        Posterior predictive draws at H_future.

        Returns:
          y_rep: (S, nF) int32
          mu:    (S, nF) float
        """
        fit = self._require_fit()
        s = fit.posterior_samples

        Hf = jnp.asarray(H_future, dtype=jnp.float32)
        beta = s["beta"]
        alpha = s["alpha"]

        S_total = beta.shape[0]
        if n_draws is None or n_draws >= S_total:
            idx = jnp.arange(S_total)
        else:
            idx = jnp.linspace(0, S_total - 1, n_draws).astype(jnp.int32)

        beta = beta[idx]
        alpha = alpha[idx]

        def _one_draw(key, beta_s, alpha_s):
            eta = Hf @ beta_s
            mu = jnp.exp(eta)
            nb = self._nb2_dist(mu=mu, alpha=alpha_s)
            y_rep = nb.sample(key)
            return y_rep.astype(jnp.int32), mu

        keys = jax.random.split(rng_key, beta.shape[0])
        y_rep, mu_rep = jax.vmap(_one_draw)(keys, beta, alpha)

        return {"y_rep": y_rep, "mu": mu_rep}

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
        """
        Hf = np.asarray(h_row, dtype=np.float32).reshape(1, -1)
        pred = self.posterior_predictive_samples(H_future=Hf, rng_key=rng_key, n_draws=n_draws)
        y_rep = pred["y_rep"]  # (S, 1)
        p = jnp.mean((y_rep[:, 0] >= int(y_obs)).astype(jnp.float32))
        return float(p)
