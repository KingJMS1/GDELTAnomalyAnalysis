"""
This script executes the TFT-Bayesian and TSMixer-Bayesian for the empirical GDELT case study. It targets 
specific CAMEO event codes (04, 13, 18, 19) across two geopolitical 
panels: Israel-Palestine (ISR) and Russia-Ukraine (UKR).

Process:
1. Loads pre-trained Temporal Fusion Transformer (TFT) and TSMixer models.
2. Extracts frozen high-dimensional embeddings for a specific target series.
3. Combines the embedding with an AR2 block (lag 1, lag 2, and intercept) to form the design matrix.
4. Fits a Bayesian likelihood head (NB2 or ZINB2, depending on sparsity)

Train/Test Split Setup:
To ensure a fair out-of-sample comparison, the test sets must start at the 
exact same chronological week regardless of the model's lookback window.
- TEST_START_ISR = 370
- TEST_START_UKR = 318
These values represent the absolute temporal indices in the raw arrays. The 
`build_design` function automatically subtracts the model-specific lookback 
offset from the training data. This means the actual 
training length varies slightly depending on the encoder's architecture, 
but the test evaluation period remains perfectly synchronized across all models.
"""

import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch as pt
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from omegaconf import OmegaConf
import GDELTAnomalies.models.tft as tft
from GDELTAnomalies.models.tsmixer import TSMixer
from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
from GDELTAnomalies.models.nb2_glm_numpyro import NB2GLM
from GDELTAnomalies.models.zinb2_glm_numpyro import ZINB2GLM

warnings.filterwarnings("ignore")
pt.set_num_threads(1)

# Configuration
NUM_WARMUP = 500
NUM_SAMPLES = 10000 
EVENT_FILTER = ["04", "13", "18", "19"]

TSM_ISR_CKPT = "checkpoints/tsmixer_isr_small/TSMixer_small_ff_dim=19_lookback=5_num_blocks=3.pt"
TFT_ISR_CKPT = "checkpoints/TFT_isr_small/TFT_isr_small_hdim_24_1300.pt"
TSM_UKR_CKPT = "checkpoints/tsmixer_ukr_small/TSMixer_ukr_small_ff_dim=48_lookback=29_num_blocks=3.pt"
TFT_UKR_CKPT = "checkpoints/TFT_ukr_small/TFT_ukr_small_hdim_24_1550.pt"

OUTPUT_DIR_METRICS = "output_samples_gdelt/metrics"
OUTPUT_DIR_ARRAYS = "output_samples_gdelt/arrays"
os.makedirs(OUTPUT_DIR_METRICS, exist_ok=True)
os.makedirs(OUTPUT_DIR_ARRAYS, exist_ok=True)

# Feature Extraction
def clean_state_dict(sd):
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

def extract_tft_single_series(ckpt_path, dataset, target_idx):
    checkpoint = pt.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    
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
    
    indices = list(range(target_idx, len(dataset), dataset.num_series))
    subset = pt.utils.data.Subset(dataset, indices)
    loader = pt.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
    
    all_embs = []
    with pt.no_grad():
        for X, y, static in loader:
            batch = {"historical_ts_numeric": X, "static_feats_numeric": static}
            model(batch)
            emb = extracted['emb'].squeeze(1).numpy()
            all_embs.append(emb)
            
    handle.remove()
    return np.concatenate(all_embs, axis=0) 

def extract_tsm_single_series(ckpt_path, dataset_unflattened, lookback, ff_dim, target_idx):
    checkpoint = pt.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])
    
    model = TSMixer(lookback=lookback, horizon=1, input_series=1120, num_blocks=3, ff_dim=ff_dim).to("cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    all_embs = []
    with pt.no_grad():
        for i in range(len(dataset_unflattened)):
            X, _ = dataset_unflattened[i]
            h = model.mixer_layers(X.unsqueeze(0)) 
            emb = h[0, :, target_idx].numpy() 
            all_embs.append(emb) 
            
    return np.array(all_embs) 

def compute_metrics(y_true, y_rep):
    y_med = np.median(y_rep, axis=0)
    mae_log = np.mean(np.abs(np.log10(y_med + 1) - np.log10(y_true + 1)))
    
    L = np.percentile(y_rep, 2.5, axis=0)
    U = np.percentile(y_rep, 97.5, axis=0)
    width_log = np.mean(np.log10(U + 1) - np.log10(L + 1))
    
    n_exceed = np.sum(y_true > U)
    tail = np.abs(0.025 - (n_exceed / len(y_true)))
    
    return float(mae_log), float(tail), float(width_log), y_med, L, U

def fit_and_eval(y_tr, H_tr, y_te, H_te):
    zero_rate = np.mean(y_tr == 0)
    density_name = "ZINB2" if zero_rate >= 0.65 else "NB2"
    ModelClass = ZINB2GLM if zero_rate >= 0.65 else NB2GLM
    
    glm = ModelClass()
    glm.fit(y=y_tr, H=H_tr, rng_key=jax.random.PRNGKey(42), num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES)
    pred = glm.posterior_predictive_samples(H_future=H_te, rng_key=jax.random.PRNGKey(99))
    
    mae, tail, width, y_med, L, U = compute_metrics(y_te, pred["y_rep"])
    return density_name, float(zero_rate), mae, tail, width, y_med, L, U

if __name__ == "__main__":
    try:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    except ValueError:
        print("Invalid SLURM_ARRAY_TASK_ID. Exiting.")
        sys.exit(1)
        
    ds_base = GDELTDataset(lookback=5, horizon=1, step=1, event_filter=EVENT_FILTER, flatten=False)
    if task_id >= ds_base.num_series:
        print(f"Task ID {task_id} exceeds series count {ds_base.num_series}. Exiting.")
        sys.exit(0)
        
    series_name = ds_base.columns[task_id]
    print(f"=== Processing Task {task_id}: {series_name} ===")

    ds_ukr = GDELTDataset(lookback=29, horizon=1, step=1, event_filter=EVENT_FILTER, flatten=False)
    ds_tft_base = GDELTDataset(lookback=10, horizon=1, step=1, event_filter=EVENT_FILTER, flatten=True)
    
    tft_isr = extract_tft_single_series(TFT_ISR_CKPT, ds_tft_base, task_id)
    tft_ukr = extract_tft_single_series(TFT_UKR_CKPT, ds_tft_base, task_id)
    tsm_isr = extract_tsm_single_series(TSM_ISR_CKPT, ds_base, 5, 19, task_id)
    tsm_ukr = extract_tsm_single_series(TSM_UKR_CKPT, ds_ukr, 29, 48, task_id)
    
    y_raw = ds_base.data[:, task_id].numpy()
    T_total = len(y_raw)
    TEST_START_ISR, TEST_START_UKR = 370, 318
    
    lag1 = np.pad(np.log1p(y_raw[:-1]), (1, 0), constant_values=0)
    lag2 = np.pad(np.log1p(y_raw[:-2]), (2, 0), constant_values=0)
    intercept = np.ones(T_total)
    
    def build_design(emb_feat, start_offset, test_start):
        H_full = np.column_stack([intercept, lag1, lag2])
        padded_emb = np.vstack([np.zeros((start_offset, emb_feat.shape[1])), emb_feat])
        H_full = np.column_stack([H_full, padded_emb])
        return y_raw[start_offset:test_start], H_full[start_offset:test_start], y_raw[test_start:], H_full[test_start:]

    y_tr_ti, H_tr_ti, y_te_i, H_te_ti = build_design(tft_isr, 10, TEST_START_ISR)
    y_tr_mi, H_tr_mi, _,      H_te_mi = build_design(tsm_isr, 5,  TEST_START_ISR)
    y_tr_tu, H_tr_tu, y_te_u, H_te_tu = build_design(tft_ukr, 10, TEST_START_UKR)
    y_tr_mu, H_tr_mu, _,      H_te_mu = build_design(tsm_ukr, 29, TEST_START_UKR)

    print("Fitting models...")
    d_ti, z_ti, mae_ti, tail_ti, w_ti, med_ti, L_ti, U_ti = fit_and_eval(y_tr_ti, H_tr_ti, y_te_i, H_te_ti)
    d_mi, z_mi, mae_mi, tail_mi, w_mi, med_mi, L_mi, U_mi = fit_and_eval(y_tr_mi, H_tr_mi, y_te_i, H_te_mi)
    d_tu, z_tu, mae_tu, tail_tu, w_tu, med_tu, L_tu, U_tu = fit_and_eval(y_tr_tu, H_tr_tu, y_te_u, H_te_tu)
    d_mu, z_mu, mae_mu, tail_mu, w_mu, med_mu, L_mu, U_mu = fit_and_eval(y_tr_mu, H_tr_mu, y_te_u, H_te_mu)
    
    metrics = {
        "Series": series_name,
        "TFT_ISR_Density": d_ti, "TFT_ISR_ZeroRate": z_ti, "TFT_ISR_MAE": mae_ti, "TFT_ISR_Tail": tail_ti, "TFT_ISR_Width": w_ti,
        "TSM_ISR_Density": d_mi, "TSM_ISR_ZeroRate": z_mi, "TSM_ISR_MAE": mae_mi, "TSM_ISR_Tail": tail_mi, "TSM_ISR_Width": w_mi,
        "TFT_UKR_Density": d_tu, "TFT_UKR_ZeroRate": z_tu, "TFT_UKR_MAE": mae_tu, "TFT_UKR_Tail": tail_tu, "TFT_UKR_Width": w_tu,
        "TSM_UKR_Density": d_mu, "TSM_UKR_ZeroRate": z_mu, "TSM_UKR_MAE": mae_mu, "TSM_UKR_Tail": tail_mu, "TSM_UKR_Width": w_mu
    }

    pd.DataFrame([metrics]).to_csv(f"{OUTPUT_DIR_METRICS}/metrics_{series_name}.csv", index=False)
    np.savez_compressed(
        f"{OUTPUT_DIR_ARRAYS}/{series_name}_preds.npz",
        y_te_isr=y_te_i, y_te_ukr=y_te_u,
        tft_isr_med=med_ti, tft_isr_L=L_ti, tft_isr_U=U_ti,
        tsm_isr_med=med_mi, tsm_isr_L=L_mi, tsm_isr_U=U_mi,
        tft_ukr_med=med_tu, tft_ukr_L=L_tu, tft_ukr_U=U_tu,
        tsm_ukr_med=med_mu, tsm_ukr_L=L_mu, tsm_ukr_U=U_mu
    )
    print(f"=== Task {task_id} completed successfully ===")