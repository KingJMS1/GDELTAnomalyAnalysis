import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class ZINB2FitResult:
    mcmc: MCMC
    posterior_samples: Dict[str, jnp.ndarray]

class ZINB2GLM_Shrinkage:
    # Applies Gaussian to H_fixed and TPBN to H_shrink for Mean (mu) and Gate (pi).
    def __init__(self, u: float = 0.5, a: float = 0.5, tau_0: float = 0.5, alpha_shape: float = 1.0, alpha_rate: float = 10.0, fixed_sigma: float = 10.0):
        self.u = float(u)
        self.a = float(a)
        self.tau_0 = float(tau_0)
        self.alpha_shape = float(alpha_shape)
        self.alpha_rate = float(alpha_rate)
        self.fixed_sigma = float(fixed_sigma)
        self.fit_result: Optional[ZINB2FitResult] = None

    def model(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        n_obs, n_fixed = H_fixed.shape
        _, n_shrink = H_shrink.shape

        # Mean Component (mu)
        # Fixed (Gaussian)
        beta_fixed = numpyro.sample("beta_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))
        
        # Shrink (TPBN)
        kappa_b = numpyro.sample("kappa_b", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_b = jnp.sqrt((1.0 - kappa_b) / (kappa_b + 1e-10))
        tau_b = numpyro.sample("tau_b", dist.HalfCauchy(self.tau_0))
        beta_shrink = numpyro.sample("beta_shrink", dist.Normal(0, tau_b * lam_b))

        # Gate Component (pi)
        # Fixed (Gaussian)
        gamma_fixed = numpyro.sample("gamma_fixed", dist.Normal(0, self.fixed_sigma).expand([n_fixed]))
        
        # Shrink (TPBN)
        kappa_g = numpyro.sample("kappa_g", dist.Beta(self.u, self.a).expand([n_shrink]))
        lam_g = jnp.sqrt((1.0 - kappa_g) / (kappa_g + 1e-10))
        tau_g = numpyro.sample("tau_g", dist.HalfCauchy(self.tau_0))
        gamma_shrink = numpyro.sample("gamma_shrink", dist.Normal(0, tau_g * lam_g))

        # Likelihood
        alpha = numpyro.sample("alpha", dist.Gamma(self.alpha_shape, self.alpha_rate))
        concentration = 1.0 / (alpha + 1e-5)

        # Compute Mu
        eta_mu = jnp.dot(H_fixed, beta_fixed) + jnp.dot(H_shrink, beta_shrink)
        mu = jnp.exp(jnp.clip(eta_mu, -15.0, 15.0))
        
        # Compute Pi
        eta_pi = jnp.dot(H_fixed, gamma_fixed) + jnp.dot(H_shrink, gamma_shrink)
        pi = jax.nn.sigmoid(eta_pi)

        if y is not None:
            numpyro.sample("obs", dist.ZeroInflatedNegativeBinomial2(mean=mu, concentration=concentration, gate=pi), obs=y)

    def fit(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, y: jnp.ndarray, 
            num_warmup=500, num_samples=1000, rng_key=None):
        if rng_key is None: rng_key = jax.random.PRNGKey(0)
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=True)
        mcmc.run(rng_key, H_fixed=H_fixed, H_shrink=H_shrink, y=y)
        self.fit_result = ZINB2FitResult(mcmc=mcmc, posterior_samples=mcmc.get_samples())
        return self.fit_result

    def get_selected_indices(self) -> List[int]:
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        s = self.fit_result.posterior_samples
        
        # Check Beta Shrink
        b_lo = jnp.percentile(s["beta_shrink"], 2.5, axis=0)
        b_hi = jnp.percentile(s["beta_shrink"], 97.5, axis=0)
        active_beta = (jnp.sign(b_lo) == jnp.sign(b_hi))
        
        # Check Gamma Shrink
        g_lo = jnp.percentile(s["gamma_shrink"], 2.5, axis=0)
        g_hi = jnp.percentile(s["gamma_shrink"], 97.5, axis=0)
        active_gamma = (jnp.sign(g_lo) == jnp.sign(g_hi))
        
        active_mask = jnp.logical_or(active_beta, active_gamma)
        return jnp.where(active_mask)[0].tolist()

    def predict(self, H_fixed: jnp.ndarray, H_shrink: jnp.ndarray, rng_key=None):
        if self.fit_result is None: raise RuntimeError("Model not fitted.")
        if rng_key is None: rng_key = jax.random.PRNGKey(1)
        
        s = self.fit_result.posterior_samples
        n_samples = s["beta_fixed"].shape[0]

        def _single_predict(k, bf, bs, gf, gs, a):
            mu = jnp.exp(jnp.clip(jnp.dot(H_fixed, bf) + jnp.dot(H_shrink, bs), -15.0, 15.0))
            pi = jax.nn.sigmoid(jnp.dot(H_fixed, gf) + jnp.dot(H_shrink, gs))
            conc = 1.0 / (a + 1e-5)
            
            k1, k2 = jax.random.split(k, 2)
            is_zero = dist.Bernoulli(pi).sample(k1)
            y_nb = dist.NegativeBinomial2(mu, conc).sample(k2)
            y_rep = jnp.where(is_zero.astype(bool), 0, y_nb)
            return y_rep, mu, pi

        keys = jax.random.split(rng_key, n_samples)
        y_rep, mu_rep, pi_rep = jax.vmap(_single_predict)(keys, s["beta_fixed"], s["beta_shrink"], 
                                                          s["gamma_fixed"], s["gamma_shrink"], s["alpha"])
        return {"y_rep": y_rep, "mu": mu_rep, "pi": pi_rep}