print("Entered")
import os
import sys
import numpy as np
import pandas as pd

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
from GDELTAnomalies.models.nb2_glm_shrinkage import NB2GLM_Shrinkage
from GDELTAnomalies.models.zinb2_glm_shrinkage import ZINB2GLM_Shrinkage

def run_batch(index, output_dir):
    EVENT_FILTER = ["04", "13", "18", "19"]
    TRAIN_WEEKS_GLM = 360

    dataset = GDELTDataset(event_filter=EVENT_FILTER)
    df = dataset.df
    print(len(df.columns))
    time_index = pd.to_datetime(df.index)

    NUM_WARMUP = 100
    NUM_SAMPLES = 20000   
    column = df.columns[index]

    weeks_idx = time_index.isocalendar().week.to_numpy()[2:] 
    X_calendar = pd.get_dummies(weeks_idx, prefix="Week", drop_first=True).astype(float).values
    
    X_all_raw = df.drop(column, axis=1).values
    X_all_ar1 = np.log1p(X_all_raw)[1:-1]

    H_shrink = np.column_stack([X_all_ar1, X_calendar])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Fitting Series: {column}")
    
    y_raw = df[column].values
    y_aligned = y_raw[2:] 
    
    t_len = len(y_aligned)
    intercept = np.ones(t_len, dtype=np.float32)
    target_lag1 = np.log1p(y_raw[1:-1])
    target_lag2 = np.log1p(y_raw[:-2])
    H_fixed = np.column_stack([intercept, target_lag1, target_lag2])
    
    y_tr = y_aligned[:TRAIN_WEEKS_GLM-2]
    H_fixed_tr = H_fixed[:TRAIN_WEEKS_GLM-2]
    H_shrink_tr = H_shrink[:TRAIN_WEEKS_GLM-2]
    
    zero_rate = (y_tr == 0).mean()
    if zero_rate > 0.65:
        ModelClass = ZINB2GLM_Shrinkage
        model_type = "ZINB2"
    else:
        ModelClass = NB2GLM_Shrinkage
        model_type = "NB2"
        
    model = ModelClass(u=0.5, a=0.5, tau_0=0.1, fixed_sigma=100.0)
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
    
    save_path = os.path.join(output_dir, f"{column}.npz")
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    index = sys.argv[1]
    print(index)
    run_batch(int(index) - 1, "glm_output")