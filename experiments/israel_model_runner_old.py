import argparse
import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import importlib
import gc
import numpyro

from GDELTAnomalies.models.nb2_glm_shrinkage import NB2GLM_Shrinkage
from GDELTAnomalies.models.zinb2_glm_shrinkage import ZINB2GLM_Shrinkage

def run_batch(start_index, batch_size, csv_path, output_dir):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    EVENT_FILTER = ["04", "13", "18", "19"]
    TRAIN_WEEKS_GLM = 360
    NUM_WARMUP = 5000
    NUM_SAMPLES = 5000
    
    relevant_cols = [c for c in df.columns if c.split("_")[1] in EVENT_FILTER]
    
    total_series = len(relevant_cols)
    end_index = min(start_index + batch_size, total_series)
    
    if start_index >= total_series:
        print(f"Start index {start_index} out of range. Exiting.")
        return

    target_cols_batch = relevant_cols[start_index : end_index]
    print(f"Processing Batch: {start_index} to {end_index-1}")

    df_small = df[relevant_cols].copy()

    weeks_idx = df_small.index.isocalendar().week.to_numpy()[2:] 
    X_calendar = pd.get_dummies(weeks_idx, prefix="Week").astype(float).values
    if X_calendar.shape[1] > 52: X_calendar = X_calendar[:, :52]
    
    X_all_raw = df_small.values
    X_all_ar1 = np.log1p(X_all_raw[1:-1]).astype(np.float32)

    H_shrink_all = np.column_stack([X_all_ar1, X_calendar]).astype(np.float32)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, target_col in enumerate(target_cols_batch):
        try:
            print(f"Fitting Series: {target_col}")
            
            y_raw = df_small[target_col].values
            y_aligned = y_raw[2:] 
            
            t_len = len(y_aligned)
            intercept = np.ones(t_len, dtype=np.float32)
            target_lag1 = np.log1p(y_raw[1:-1])
            target_lag2 = np.log1p(y_raw[:-2])
            H_fixed = np.column_stack([intercept, target_lag1, target_lag2]).astype(np.float32)
            
            H_shrink_current = H_shrink_all.copy()
            target_feat_idx = df_small.columns.get_loc(target_col)
            H_shrink_current[:, target_feat_idx] = 0.0 
            
            y_tr = y_aligned[:TRAIN_WEEKS_GLM-2]
            H_fixed_tr = H_fixed[:TRAIN_WEEKS_GLM-2]
            H_shrink_tr = H_shrink_current[:TRAIN_WEEKS_GLM-2]
            
            zero_rate = (y_tr == 0).mean()
            if zero_rate > 0.65:
                ModelClass = ZINB2GLM_Shrinkage
                model_type = "ZINB2"
            else:
                ModelClass = NB2GLM_Shrinkage
                model_type = "NB2"
                
            model = ModelClass(u=0.5, a=0.5, tau_0=0.1, fixed_sigma=10.0)
            model.fit(
                H_fixed=H_fixed_tr, 
                H_shrink=H_shrink_tr, 
                y=y_tr, 
                num_warmup=NUM_WARMUP, 
                num_samples=NUM_SAMPLES
            )
            
            s = model.fit_result.posterior_samples
            save_dict = {k: np.array(v) for k, v in s.items()}
            
            selected_indices = model.get_selected_indices()
            
            save_dict['selected_indices'] = np.array(selected_indices)
            save_dict['model_type'] = model_type
            
            save_path = os.path.join(output_dir, f"{target_col}.npz")
            np.savez_compressed(save_path, **save_dict)
            print(f"Saved: {save_path}")

            del model
            del H_shrink_current
            del H_fixed
            del s
            del save_dict
            del y_tr
            del H_fixed_tr
            del H_shrink_tr
            gc.collect()

        except Exception as e:
            print(f"Error fitting {target_col}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--csv_path", type=str, default="gdelt_fix.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    run_batch(args.start_index, args.batch_size, args.csv_path, args.output_dir)