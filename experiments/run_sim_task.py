"""
Simulation Experiment

Task IDs:
    0: AR2 Baseline GLM
    1: Full GLM
    2: TFT-Bayesian Head
    3: TSMixer-Bayesian Head
    4: Two-Step Shrinkage GLM
    5: TSMixer + Shrinkage
    6: True DGP (Oracle)

Outputs are saved as .npy files in the designated output directory for subsequent
evaluation and visualization.
"""

import sys
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import torch as pt
import numpy as np
import jax
import numpyro
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from scipy.stats import nbinom

from GDELTAnomalies.datasets.sim_dataset import SimDataset
import GDELTAnomalies.models.tft as tft
from GDELTAnomalies.models.tsmixer import TSMixer
from GDELTAnomalies.models.nb2_glm_numpyro import NB2GLM
from GDELTAnomalies.models.zinb2_glm_numpyro import ZINB2GLM
from GDELTAnomalies.models.glm_shrinkage_models import NB2GLM_Shrinkage, ZINB2GLM_Shrinkage


warnings.filterwarnings("ignore")
pt.set_num_threads(1)
numpyro.set_host_device_count(1)

NUM_WARMUP = 1000
NUM_SAMPLES = 6000
TRAIN_LEN = 950
TRUE_ACTIVE_INDICES = [0, 1, 2, 3, 4]

TSM_CKPT = "checkpoints/TSMixer_sim/TSMixer_sim_ff_dim=24_lookback=10_num_blocks=3_epoch=7000.pt"
TFT_CKPT = "checkpoints/TFT_sim/TFT_hdim_24_500.pt"
OUTPUT_DIR = "sim_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_state_dict(sd):
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

def extract_tft_sim(ckpt_path, dataset):
    checkpoint = pt.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint.get("model_state_dict", checkpoint))
    config = OmegaConf.create({
        'model': {'dropout': 0.05, 'state_size': 24, 'output_quantiles': [0.5], 'lstm_layers': 2, 'attention_heads': 3},
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': {
            'num_historical_numeric': dataset.num_series, 'num_historical_categorical': 0,
            'num_static_numeric': dataset.num_statics, 'num_static_categorical': 0,
            'num_future_numeric': 0, 'num_future_categorical': 0,
            'num_future_steps': 1, 'device': 'cpu'
        }
    })
    model = tft.TemporalFusionTransformer(config).to("cpu")
    model.load_state_dict(state_dict)
    model.eval()
    extracted = {}
    def hook(module, inp, out):
        extracted['emb'] = out.detach()
    handle = model.pos_wise_ff_gating.register_forward_hook(hook)
    loader = pt.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    all_embs = []
    with pt.no_grad():
        for X, y, static in loader:
            batch = {"historical_ts_numeric": X, "static_feats_numeric": static}
            model(batch)
            all_embs.append(extracted['emb'].squeeze(1).numpy())
    handle.remove()
    all_embs = np.concatenate(all_embs, axis=0) 
    return all_embs.reshape(dataset.num_times, dataset.num_series, 24)

def extract_tsm_sim(ckpt_path, dataset_unflattened):
    checkpoint = pt.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model = TSMixer(lookback=10, horizon=1, input_series=100, output_series=2, num_blocks=3, ff_dim=24).to("cpu")
    model.load_state_dict(state_dict)
    model.eval()
    all_embs = []
    with pt.no_grad():
        for i in range(len(dataset_unflattened)):
            X, _ = dataset_unflattened[i]
            h = model.mixer_layers(X.unsqueeze(0)) 
            all_embs.append(h[0].numpy().T) 
    return np.array(all_embs)

def compute_metrics(y_true, y_med, L, U):
    mae_log = np.mean(np.abs(np.log10(y_med + 1) - np.log10(y_true + 1)))
    mae_raw = np.mean(np.abs(y_med - y_true))
    n_exceed = np.sum(y_true > U)
    tail = np.abs(0.025 - (n_exceed / len(y_true)))
    return float(mae_raw), float(mae_log), float(tail)

def run_standard_model(ModelClass, H_tr, y_tr, H_te, y_te, chains=1):
    glm = ModelClass()
    glm.fit(y=y_tr, H=H_tr, rng_key=jax.random.PRNGKey(42), num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=chains)
    pred = glm.posterior_predictive_samples(H_future=H_te, rng_key=jax.random.PRNGKey(99))
    y_rep = pred["y_rep"]
    y_med = np.median(y_rep, axis=0)
    L = np.percentile(y_rep, 2.5, axis=0)
    U = np.percentile(y_rep, 97.5, axis=0)
    return y_med, L, U, np.nan

def run_twostep_model(ModelClass, H_f_tr, H_s_tr, y_tr, H_f_te, H_s_te, y_te, calc_f1=True, chains=1):
    glm = ModelClass()
    glm.fit(H_fixed=H_f_tr, H_shrink=H_s_tr, y=y_tr, rng_key=jax.random.PRNGKey(42), num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=chains)
    active_idx = glm.get_selected_indices()
    
    if calc_f1:
        y_pred_binary = np.zeros(H_s_tr.shape[1])
        if len(active_idx) > 0:
            y_pred_binary[active_idx] = 1
        y_true_binary = np.zeros(H_s_tr.shape[1])
        y_true_binary[TRUE_ACTIVE_INDICES] = 1
        f1 = float(f1_score(y_true_binary, y_pred_binary))
    else:
        f1 = np.nan
        
    H_s_tr_active = H_s_tr[:, active_idx] if len(active_idx) > 0 else np.empty((H_s_tr.shape[0], 0))
    H_s_te_active = H_s_te[:, active_idx] if len(active_idx) > 0 else np.empty((H_s_te.shape[0], 0))
    
    glm_step2 = ModelClass()
    glm_step2.fit(H_fixed=H_f_tr, H_shrink=H_s_tr_active, y=y_tr, rng_key=jax.random.PRNGKey(101), num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=chains)
    pred = glm_step2.predict(H_fixed=H_f_te, H_shrink=H_s_te_active, rng_key=jax.random.PRNGKey(99))
    y_rep = pred["y_rep"]
    y_med = np.median(y_rep, axis=0)
    L = np.percentile(y_rep, 2.5, axis=0)
    U = np.percentile(y_rep, 97.5, axis=0)
    return y_med, L, U, f1

def run_true_model(H_f_te_d, H_s_te_d, H_f_te_s, H_s_te_s):
    alpha_true = 0.5
    concentration = 1.0 / alpha_true
    
    beta_fixed_d = np.array([0.5, 0.2, 0.1]) 
    beta_shrink_d = np.zeros(99)
    beta_shrink_d[0] = 0.4          
    beta_shrink_d[1:5] = np.array([0.6, -0.5, 0.5, -0.6]) 
    
    eta_d = np.dot(H_f_te_d, beta_fixed_d) + np.dot(H_s_te_d, beta_shrink_d)
    mu_d = np.exp(np.clip(eta_d, -15.0, 15.0))
    
    p_d = concentration / (concentration + mu_d)
    med_d = nbinom.ppf(0.5, concentration, p_d)
    L_d = nbinom.ppf(0.025, concentration, p_d)
    U_d = nbinom.ppf(0.975, concentration, p_d)
    
    beta_fixed_s = np.array([0.2, 0.1, 0.05])
    gamma_fixed_s = np.array([-1.0, 0.2, 0.1]) 
    beta_shrink_s = np.zeros(99)
    beta_shrink_s[0] = 0.5          
    beta_shrink_s[1:5] = np.array([-0.6, 0.4, -0.5, 0.6]) 
    gamma_shrink_s = np.zeros(99)
    gamma_shrink_s[0] = 0.6         
    gamma_shrink_s[1:5] = np.array([0.7, -0.5, 0.6, -0.7]) 
    
    eta_mu_s = np.dot(H_f_te_s, beta_fixed_s) + np.dot(H_s_te_s, beta_shrink_s)
    eta_pi_s = np.dot(H_f_te_s, gamma_fixed_s) + np.dot(H_s_te_s, gamma_shrink_s)
    mu_s = np.exp(np.clip(eta_mu_s, -15.0, 15.0))
    pi_s = np.clip(1.0 / (1.0 + np.exp(-eta_pi_s)), 1e-6, 1.0 - 1e-6)
    
    p_s = concentration / (concentration + mu_s)
    samples_s = np.where(
        np.random.binomial(1, pi_s, size=(10000, len(mu_s))) == 1,
        0,
        np.random.negative_binomial(concentration, p_s, size=(10000, len(mu_s)))
    )
    med_s = np.median(samples_s, axis=0)
    L_s = np.percentile(samples_s, 2.5, axis=0)
    U_s = np.percentile(samples_s, 97.5, axis=0)
    
    return (med_d, L_d, U_d), (med_s, L_s, U_s)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sim_task.py <TASK_ID>")
        sys.exit(1)
        
    task_id = int(sys.argv[1])
    
    models_map = {
        0: "AR2_GLM",
        1: "Full_GLM",
        2: "TFT_GLM",
        3: "TSM_GLM",
        4: "Two-Step",
        5: "TSM_Shrinkage",
        6: "True_Model"
    }
    
    if task_id not in models_map:
        print(f"Invalid Task ID: {task_id}")
        sys.exit(1)
        
    model_name = models_map[task_id]
    print(f"Executing Task {task_id}: {model_name}")
    
    ds_base = SimDataset(lookback=10, horizon=1, step=1, flatten=False)
    y_all = ds_base.data.numpy() 
    y_dense = y_all[:, 0]
    y_sparse = y_all[:, 1]

    lag1_dense = np.pad(np.log1p(y_dense[:-1]), (1, 0), constant_values=0)
    lag2_dense = np.pad(np.log1p(y_dense[:-2]), (2, 0), constant_values=0)
    lag1_sparse = np.pad(np.log1p(y_sparse[:-1]), (1, 0), constant_values=0)
    lag2_sparse = np.pad(np.log1p(y_sparse[:-2]), (2, 0), constant_values=0)

    x_cov_all = y_all[:, 2:] 
    x_cov_lag1 = np.pad(np.log1p(x_cov_all[:-1, :]), ((1, 0), (0, 0)), constant_values=0)
    intercept = np.ones((len(y_all), 1))

    H_fix_dense = np.column_stack([intercept, lag1_dense, lag2_dense])
    H_shk_dense = np.column_stack([lag1_sparse, x_cov_lag1]) 
    H_fix_sparse = np.column_stack([intercept, lag1_sparse, lag2_sparse])
    H_shk_sparse = np.column_stack([lag1_dense, x_cov_lag1])
    
    if task_id == 6:
        (med_d, L_d, U_d), (med_s, L_s, U_s) = run_true_model(
            H_fix_dense[TRAIN_LEN:], H_shk_dense[TRAIN_LEN:], 
            H_fix_sparse[TRAIN_LEN:], H_shk_sparse[TRAIN_LEN:]
        )
        f1_d, f1_s = np.nan, np.nan
        
    else:
        if task_id in [2, 3, 5]:
            ds_tft = SimDataset(lookback=10, horizon=1, step=1, flatten=True)
            tft_emb = extract_tft_sim(TFT_CKPT, ds_tft)
            tsm_emb = extract_tsm_sim(TSM_CKPT, ds_base)
            t_emb_pad = np.vstack([np.zeros((10, tft_emb.shape[2])), tft_emb[:, 0, :]])
            tsm_emb_pad_d = np.vstack([np.zeros((10, tsm_emb.shape[2])), tsm_emb[:, 0, :]])
            tsm_emb_pad_s = np.vstack([np.zeros((10, tsm_emb.shape[2])), tsm_emb[:, 1, :]])
            
            H_tr_tft_d = np.column_stack([H_fix_dense, t_emb_pad])[10:TRAIN_LEN]
            H_te_tft_d = np.column_stack([H_fix_dense, t_emb_pad])[TRAIN_LEN:]
            H_tr_tft_s = np.column_stack([H_fix_sparse, np.vstack([np.zeros((10, tft_emb.shape[2])), tft_emb[:, 1, :]])])[10:TRAIN_LEN]
            H_te_tft_s = np.column_stack([H_fix_sparse, np.vstack([np.zeros((10, tft_emb.shape[2])), tft_emb[:, 1, :]])])[TRAIN_LEN:]
            
            H_tr_tsm_d = np.column_stack([H_fix_dense, tsm_emb_pad_d])[10:TRAIN_LEN]
            H_te_tsm_d = np.column_stack([H_fix_dense, tsm_emb_pad_d])[TRAIN_LEN:]
            H_tr_tsm_s = np.column_stack([H_fix_sparse, tsm_emb_pad_s])[10:TRAIN_LEN]
            H_te_tsm_s = np.column_stack([H_fix_sparse, tsm_emb_pad_s])[TRAIN_LEN:]
            
            H_tr_tsm_fix_d = np.column_stack([H_fix_dense, tsm_emb_pad_d])[10:TRAIN_LEN]
            H_te_tsm_fix_d = np.column_stack([H_fix_dense, tsm_emb_pad_d])[TRAIN_LEN:]
            H_tr_tsm_fix_s = np.column_stack([H_fix_sparse, tsm_emb_pad_s])[10:TRAIN_LEN]
            H_te_tsm_fix_s = np.column_stack([H_fix_sparse, tsm_emb_pad_s])[TRAIN_LEN:]

        y_tr_d = y_dense[10:TRAIN_LEN]
        y_te_d = y_dense[TRAIN_LEN:]
        y_tr_s = y_sparse[10:TRAIN_LEN]
        y_te_s = y_sparse[TRAIN_LEN:]
        
        chains = 1

        if task_id == 0: 
            med_d, L_d, U_d, f1_d = run_standard_model(NB2GLM, H_fix_dense[10:TRAIN_LEN], y_tr_d, H_fix_dense[TRAIN_LEN:], y_te_d, chains)
            med_s, L_s, U_s, f1_s = run_standard_model(ZINB2GLM, H_fix_sparse[10:TRAIN_LEN], y_tr_s, H_fix_sparse[TRAIN_LEN:], y_te_s, chains)
        elif task_id == 1: 
            H_full_tr_d = np.column_stack([H_fix_dense, H_shk_dense])[10:TRAIN_LEN]
            H_full_te_d = np.column_stack([H_fix_dense, H_shk_dense])[TRAIN_LEN:]
            med_d, L_d, U_d, f1_d = run_standard_model(NB2GLM, H_full_tr_d, y_tr_d, H_full_te_d, y_te_d, chains)
            H_full_tr_s = np.column_stack([H_fix_sparse, H_shk_sparse])[10:TRAIN_LEN]
            H_full_te_s = np.column_stack([H_fix_sparse, H_shk_sparse])[TRAIN_LEN:]
            med_s, L_s, U_s, f1_s = run_standard_model(ZINB2GLM, H_full_tr_s, y_tr_s, H_full_te_s, y_te_s, chains)
        elif task_id == 2: 
            med_d, L_d, U_d, f1_d = run_standard_model(NB2GLM, H_tr_tft_d, y_tr_d, H_te_tft_d, y_te_d, chains)
            med_s, L_s, U_s, f1_s = run_standard_model(ZINB2GLM, H_tr_tft_s, y_tr_s, H_te_tft_s, y_te_s, chains)
        elif task_id == 3: 
            med_d, L_d, U_d, f1_d = run_standard_model(NB2GLM, H_tr_tsm_d, y_tr_d, H_te_tsm_d, y_te_d, chains)
            med_s, L_s, U_s, f1_s = run_standard_model(ZINB2GLM, H_tr_tsm_s, y_tr_s, H_te_tsm_s, y_te_s, chains)
        elif task_id == 4: 
            med_d, L_d, U_d, f1_d = run_twostep_model(NB2GLM_Shrinkage, H_fix_dense[10:TRAIN_LEN], H_shk_dense[10:TRAIN_LEN], y_tr_d, H_fix_dense[TRAIN_LEN:], H_shk_dense[TRAIN_LEN:], y_te_d, True, chains)
            med_s, L_s, U_s, f1_s = run_twostep_model(ZINB2GLM_Shrinkage, H_fix_sparse[10:TRAIN_LEN], H_shk_sparse[10:TRAIN_LEN], y_tr_s, H_fix_sparse[TRAIN_LEN:], H_shk_sparse[TRAIN_LEN:], y_te_s, True, chains)
        elif task_id == 5: 
            med_d, L_d, U_d, f1_d = run_twostep_model(NB2GLM_Shrinkage, H_tr_tsm_fix_d, H_shk_dense[10:TRAIN_LEN], y_tr_d, H_te_tsm_fix_d, H_shk_dense[TRAIN_LEN:], y_te_d, True, chains)
            med_s, L_s, U_s, f1_s = run_twostep_model(ZINB2GLM_Shrinkage, H_tr_tsm_fix_s, H_shk_sparse[10:TRAIN_LEN], y_tr_s, H_te_tsm_fix_s, H_shk_sparse[TRAIN_LEN:], y_te_s, True, chains)

    y_te_d = y_dense[TRAIN_LEN:]
    y_te_s = y_sparse[TRAIN_LEN:]
    
    raw_d, log_d, tail_d = compute_metrics(y_te_d, med_d, L_d, U_d)
    raw_s, log_s, tail_s = compute_metrics(y_te_s, med_s, L_s, U_s)
    
    output_data = {
        "model_name": model_name,
        "dense": {"med": med_d, "L": L_d, "U": U_d, "MAE_raw": raw_d, "MAE_log": log_d, "Tail": tail_d, "F1": f1_d},
        "sparse": {"med": med_s, "L": L_s, "U": U_s, "MAE_raw": raw_s, "MAE_log": log_s, "Tail": tail_s, "F1": f1_s}
    }
    
    np.save(os.path.join(OUTPUT_DIR, f"task_{task_id}_{model_name}.npy"), output_data, allow_pickle=True)
    print(f"Task {task_id} complete. Output saved.")

if __name__ == "__main__":
    main()