"""
Step 2: Unbiased Refitting for Two-Step Shrinkage GLM

This script loads the active variables selected in Step 1 and refits the Bayesian GLM. 
This provides calibrated posterior predictive intervals without excessive shrinkage bias.

To adapt this script for the Israel (ISR) panel:
1. Change `TRAIN_WEEKS_GLM` from 308 to 370.
2. Change `step1_output_dir` to "glm_output_isr".
3. Change `step2_output_dir` to "glm_refit_output_isr".
"""

import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
from GDELTAnomalies.models.glm_shrinkage_models import NB2GLM_Shrinkage, ZINB2GLM_Shrinkage

def run_refit_batch(index, step1_output_dir, step2_output_dir):
    EVENT_FILTER = ["04", "13", "18", "19"]
    TRAIN_WEEKS_GLM = 318  # 370 for ISR
    
    NUM_WARMUP = 1000
    NUM_SAMPLES = 20000 
    
    dataset = GDELTDataset(event_filter=EVENT_FILTER)
    df = dataset.df
    
    if index >= len(df.columns):
        print(f"Error: Index {index} out of bounds.")
        return

    column = df.columns[index]
    
    step1_file = os.path.join(step1_output_dir, f"{column}.npz")
    
    if not os.path.exists(step1_file):
        print(f"Skipping {column}: Step 1 output not found at {step1_file}")
        return

    print(f"Processing Step 2 for: {column}")
    try:
        prev_data = np.load(step1_file, allow_pickle=True)
        selected_indices = prev_data['selected_indices']
        model_type = str(prev_data['model_type'])
    except Exception as e:
        print(f"Error reading .npz file: {e}")
        return
    
    if len(selected_indices) == 0:
        print(f"Skipping {column}: No variables selected in Step 1.")
        return

    print(f"Model Type: {model_type}")
    print(f"Selected Variables Count: {len(selected_indices)}")

    time_index = pd.to_datetime(df.index)
    weeks_idx = time_index.isocalendar().week.to_numpy()[2:] 
    X_calendar = pd.get_dummies(weeks_idx, prefix="Week", drop_first=True).astype(float).values
    
    X_all_raw = df.drop(column, axis=1).values
    X_all_ar1 = np.log1p(X_all_raw)[1:-1]

    H_shrink_full = np.column_stack([X_all_ar1, X_calendar])
    
    y_raw = df[column].values
    y_aligned = y_raw[2:] 
    t_len = len(y_aligned)
    
    intercept = np.ones(t_len, dtype=np.float32)
    target_lag1 = np.log1p(y_raw[1:-1])
    target_lag2 = np.log1p(y_raw[:-2])
    H_fixed_step2 = np.column_stack([intercept, target_lag1, target_lag2]).astype(np.float32)

    H_shrink_step2 = H_shrink_full[:, selected_indices].astype(np.float32)

    y_tr = y_aligned[:TRAIN_WEEKS_GLM-2]
    H_fixed_tr = H_fixed_step2[:TRAIN_WEEKS_GLM-2]
    H_shrink_tr = H_shrink_step2[:TRAIN_WEEKS_GLM-2]

    if not os.path.exists(step2_output_dir):
        os.makedirs(step2_output_dir, exist_ok=True)

    if model_type == "ZINB2":
        ModelClass = ZINB2GLM_Shrinkage
    else:
        ModelClass = NB2GLM_Shrinkage
    
    model = ModelClass(u=0.5, a=0.5, tau_0=0.5, fixed_sigma=100.0)
    
    model.fit(
        H_fixed=H_fixed_tr, 
        H_shrink=H_shrink_tr, 
        y=y_tr, 
        num_warmup=NUM_WARMUP, 
        num_samples=NUM_SAMPLES
    )
    
    s = model.fit_result.posterior_samples
    save_dict = {k: np.array(v) for k, v in s.items()}
    save_dict['model_type'] = model_type
    save_dict['used_indices'] = selected_indices
    
    save_path = os.path.join(step2_output_dir, f"{column}.npz")
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved Refit: {save_path}")

if __name__ == "__main__":
    index = int(sys.argv[1])
    # run_refit_batch(index - 1, "glm_output_isr", "glm_refit_output_isr")
    run_refit_batch(index - 1, "glm_output_ukr", "glm_refit_output_ukr")