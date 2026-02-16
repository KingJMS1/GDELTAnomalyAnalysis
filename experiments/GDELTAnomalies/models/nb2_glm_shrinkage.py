import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class NB2FitResult:
    mcmc: MCMC
    posterior_samples: Dict[str, jnp.ndarray]

class NB2GLM_Shrinkage:
    def __init__(self, u: float = 0.5, a: float = 0.5, tau_0: float = 0.5, alpha_shape: float = 1.0, alpha_rate: float = 10.0, fixed_sigma: float = 10.0):
        self.u = float(u)
        self.a = float(a)
        self.tau_0 = float(tau_0)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.fixed_sigma = float(fixed_sigma) # Variance for Gaussian prior
        self.fit_result: Optional[NB2FitResult] = None

    def model(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        # H_fixed:  Shape (N, p1) -> Corresponds to Beta (Gaussian)
        # H_shrink: Shape (N, p2) -> Corresponds to Gamma (TPBN)
        
        n_obs, n_fixed = H_fixed.shape
        _, n_shrink = H_shrink.shape

        # Fixed Coefficients (Gaussian Prior)
        beta_fixed = numpyro.sample("beta_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))

        # Shrinkage Coefficients (TPBN Prior)
        #    kappa ~ Beta(u, a) -> lambda^2 = (1-k)/k
        kappa = numpyro.sample("kappa", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_sq = (1.0 - kappa) / (kappa + 1e-10)
        lam = jnp.sqrt(lam_sq)
        
        tau = numpyro.sample("tau", dist.HalfCauchy(self.tau_0))
        sigma_shrink = numpyro.deterministic("sigma_shrink", tau * lam)
        
        beta_shrink = numpyro.sample("beta_shrink", dist.Normal(0, sigma_shrink))

        # Dispersion
        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))
        concentration = 1.0 / (alpha + 1e-5)

        # Linear Predictor (Combine Fixed + Shrink)
        eta = jnp.dot(H_fixed, beta_fixed) + jnp.dot(H_shrink, beta_shrink)
        mu = jnp.exp(jnp.clip(eta, -15.0, 15.0))

        if y is not None:
            numpyro.sample("obs", dist.NegativeBinomial2(mean=mu, concentration=concentration), obs=y)

    def fit(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: jnp.ndarray, 
            num_warmup=500, num_samples=1000, rng_key=None):
        if rng_key is None: rng_key = jax.random.PRNGKey(0)
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=True)
        mcmc.run(rng_key, H_fixed=H_fixed, H_shrink=H_shrink, y=y)
        self.fit_result = NB2FitResult(mcmc=mcmc, posterior_samples=mcmc.get_samples())
        return self.fit_result

    def get_selected_indices(self) -> List[int]:
        """Returns indices of H_shrink that are selected."""
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        beta_samples = self.fit_result.posterior_samples["beta_shrink"]
        lower = jnp.percentile(beta_samples, 2.5, axis=0)
        upper = jnp.percentile(beta_samples, 97.5, axis=0)
        active_mask = (jnp.sign(lower) == jnp.sign(upper))
        return jnp.where(active_mask)[0].tolist()

    def predict(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, rng_key=None):
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        if rng_key is None: rng_key = jax.random.PRNGKey(1)
        
        s = self.fit_result.posterior_samples
        b_fix = s["beta_fixed"]
        b_shr = s["beta_shrink"]
        alpha = s["alpha"]
        n_samples = b_fix.shape[0]

        def _single_predict(k, bf, bs, a):
            eta = jnp.dot(H_fixed, bf) + jnp.dot(H_shrink, bs)
            mu = jnp.exp(jnp.clip(eta, -15.0, 15.0))
            conc = 1.0 / (a + 1e-5)
            y_sample = dist.NegativeBinomial2(mu, conc).sample(k)
            return y_sample, mu

        keys = jax.random.split(rng_key, n_samples)
        y_rep, mu_rep = jax.vmap(_single_predict)(keys, b_fix, b_shr, alpha)
        return {"y_rep": y_rep, "mu": mu_rep}