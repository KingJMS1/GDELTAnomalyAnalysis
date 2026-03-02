import os
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from dataclasses import dataclass
from typing import Dict, Optional, List

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
numpyro.set_host_device_count(4)

@dataclass
class FitResult:
    mcmc: MCMC
    posterior_samples: Dict[str, jnp.ndarray]

class NB2GLM_Shrinkage:
    def __init__(self, u: float = 0.5, a: float = 0.5, tau_0: float = 0.5, alpha_shape: float = 1.0, alpha_rate: float = 10.0, fixed_sigma: float = 100.0):
        self.u = float(u)
        self.a = float(a)
        self.tau_0 = float(tau_0)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.fixed_sigma = float(fixed_sigma)
        self.fit_result: Optional[FitResult] = None
        self.n_obs: int = 0

    def model(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        n_obs, n_fixed = H_fixed.shape
        _, n_shrink = H_shrink.shape

        beta_fixed = numpyro.sample("beta_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))

        kappa = numpyro.sample("kappa", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_sq = (1.0 - kappa) / (kappa + 1e-10)
        lam = jnp.sqrt(lam_sq)
        
        tau = numpyro.sample("tau", dist.HalfCauchy(self.tau_0))
        sigma_shrink = numpyro.deterministic("sigma_shrink", tau * lam)
        beta_shrink = numpyro.sample("beta_shrink", dist.Normal(0, sigma_shrink))

        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))
        concentration = 1.0 / (alpha + 1e-5)

        eta = jnp.dot(H_fixed, beta_fixed) + jnp.dot(H_shrink, beta_shrink)
        mu = jnp.exp(jnp.clip(eta, -12.0, 10.0))

        if y is not None:
            numpyro.sample("obs", dist.NegativeBinomial2(mean=mu, concentration=concentration), obs=y)

    def fit(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: jnp.ndarray, 
            num_warmup=500, num_samples=5000, num_chains=4, rng_key=None):
        if rng_key is None: rng_key = jax.random.PRNGKey(0)
        self.n_obs = H_fixed.shape[0]
        
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                    num_chains=num_chains, chain_method='parallel', progress_bar=True)
        mcmc.run(rng_key, H_fixed=H_fixed, H_shrink=H_shrink, y=y)
        self.fit_result = FitResult(mcmc=mcmc, posterior_samples=mcmc.get_samples())
        return self.fit_result

    def get_selected_indices(self, gamma: float = 0.05) -> List[int]:
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        beta_samples = self.fit_result.posterior_samples["beta_shrink"]
        
        lower = jnp.percentile(beta_samples, 2.5, axis=0)
        upper = jnp.percentile(beta_samples, 97.5, axis=0)
        median = jnp.percentile(beta_samples, 50, axis=0)
        
        active_mask = (lower > -gamma) | (upper < gamma)
        A_n_indices = jnp.where(active_mask)[0]
        
        if len(A_n_indices) == 0:
            return []
            
        K_n = min(self.n_obs - 1, len(A_n_indices))
        K_n = max(1, K_n) 
        
        abs_med = jnp.abs(median[A_n_indices])
        sorted_idx = jnp.argsort(-abs_med)
        top_k_indices = A_n_indices[sorted_idx[:K_n]]
        
        return top_k_indices.tolist()

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
            mu = jnp.exp(jnp.clip(eta, -12.0, 10.0))
            conc = 1.0 / (a + 1e-5)
            y_sample = dist.NegativeBinomial2(mu, conc).sample(k)
            return y_sample, mu

        keys = jax.random.split(rng_key, n_samples)
        y_rep, mu_rep = jax.vmap(_single_predict)(keys, b_fix, b_shr, alpha)
        return {"y_rep": y_rep, "mu": mu_rep}


class ZINB2GLM_Shrinkage:
    def __init__(self, u: float = 0.5, a: float = 0.5, tau_0: float = 0.5, alpha_shape: float = 1.0, alpha_rate: float = 10.0, fixed_sigma: float = 100.0):
        self.u = float(u)
        self.a = float(a)
        self.tau_0 = float(tau_0)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.fixed_sigma = float(fixed_sigma)
        self.fit_result: Optional[FitResult] = None
        self.n_obs: int = 0

    def model(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        n_obs, n_fixed = H_fixed.shape
        _, n_shrink = H_shrink.shape

        beta_fixed = numpyro.sample("beta_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))
        kappa_b = numpyro.sample("kappa_b", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_b = jnp.sqrt((1.0 - kappa_b) / (kappa_b + 1e-10))
        tau_b = numpyro.sample("tau_b", dist.HalfCauchy(self.tau_0))
        beta_shrink = numpyro.sample("beta_shrink", dist.Normal(0, tau_b * lam_b))

        gamma_fixed = numpyro.sample("gamma_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))
        kappa_g = numpyro.sample("kappa_g", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_g = jnp.sqrt((1.0 - kappa_g) / (kappa_g + 1e-10))
        tau_g = numpyro.sample("tau_g", dist.HalfCauchy(self.tau_0))
        gamma_shrink = numpyro.sample("gamma_shrink", dist.Normal(0, tau_g * lam_g))

        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))
        concentration = 1.0 / (alpha + 1e-5)

        eta_mu = jnp.dot(H_fixed, beta_fixed) + jnp.dot(H_shrink, beta_shrink)
        mu = jnp.exp(jnp.clip(eta_mu, -12.0, 10.0))
        
        eta_pi = jnp.dot(H_fixed, gamma_fixed) + jnp.dot(H_shrink, gamma_shrink)
        pi = jax.nn.sigmoid(eta_pi)

        if y is not None:
            numpyro.sample("obs", dist.ZeroInflatedNegativeBinomial2(mean=mu, concentration=concentration, gate=pi), obs=y)

    def fit(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: jnp.ndarray, 
            num_warmup=500, num_samples=5000, num_chains=4, rng_key=None):
        if rng_key is None: rng_key = jax.random.PRNGKey(0)
        self.n_obs = H_fixed.shape[0]
        
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                    num_chains=num_chains, chain_method='parallel', progress_bar=True)
        mcmc.run(rng_key, H_fixed=H_fixed, H_shrink=H_shrink, y=y)
        self.fit_result = FitResult(mcmc=mcmc, posterior_samples=mcmc.get_samples())
        return self.fit_result

    def get_selected_indices(self, gamma: float = 0.05) -> List[int]:
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        s = self.fit_result.posterior_samples
        
        b_lo = jnp.percentile(s["beta_shrink"], 2.5, axis=0)
        b_hi = jnp.percentile(s["beta_shrink"], 97.5, axis=0)
        b_med = jnp.percentile(s["beta_shrink"], 50, axis=0)
        
        g_lo = jnp.percentile(s["gamma_shrink"], 2.5, axis=0)
        g_hi = jnp.percentile(s["gamma_shrink"], 97.5, axis=0)
        g_med = jnp.percentile(s["gamma_shrink"], 50, axis=0)
        
        max_lower = jnp.maximum(b_lo, g_lo)
        min_upper = jnp.minimum(b_hi, g_hi)
        
        active_mask = (max_lower > -gamma) | (min_upper < gamma)
        A_n_indices = jnp.where(active_mask)[0]
        
        if len(A_n_indices) == 0:
            return []
            
        max_abs_med = jnp.maximum(jnp.abs(b_med), jnp.abs(g_med))
        abs_med_A_n = max_abs_med[A_n_indices]
        
        K_n = min(self.n_obs - 1, len(A_n_indices))
        K_n = max(1, K_n)
        
        sorted_idx = jnp.argsort(-abs_med_A_n)
        top_k_indices = A_n_indices[sorted_idx[:K_n]]
        
        return top_k_indices.tolist()

    def predict(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, rng_key=None):
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        if rng_key is None: rng_key = jax.random.PRNGKey(1)
        
        s = self.fit_result.posterior_samples
        n_samples = s["beta_fixed"].shape[0]

        def _single_predict(k, bf, bs, gf, gs, a):
            mu = jnp.exp(jnp.clip(jnp.dot(H_fixed, bf) + jnp.dot(H_shrink, bs), -12.0, 10.0))
            pi = jax.nn.sigmoid(jnp.dot(H_fixed, gf) + jnp.dot(H_shrink, gs))
            conc = 1.0 / (a + 1e-5)
            
            k1, k2 = jax.random.split(k, 2)
            is_zero = dist.Bernoulli(pi).sample(k1)
            y_nb = dist.NegativeBinomial2(mu, conc).sample(k2)
            y_rep = jnp.where(is_zero.astype(bool), 0, y_nb)
            return y_rep, mu, pi

        keys = jax.random.split(rng_key, n_samples)
        y_rep, mu_rep, pi_rep = jax.vmap(_single_predict)(keys, s["beta_fixed"], s["beta_shrink"], s["gamma_fixed"], s["gamma_shrink"], s["alpha"])
        return {"y_rep": y_rep, "mu": mu_rep, "pi": pi_rep}