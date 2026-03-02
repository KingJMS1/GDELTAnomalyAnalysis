import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GDELTAnomalies.datasets.sim_dataset import SimDataset

OUTPUT_DIR = "sim_outputs"
TRAIN_LEN = 950

def load_experiment_results(output_dir):
    npy_files = glob.glob(os.path.join(output_dir, "*.npy"))
    
    if not npy_files:
        print(f"Warning: No .npy files found in '{output_dir}'.")
        return {}, {}, pd.DataFrame()

    plot_data_dense = {}
    plot_data_sparse = {}
    metrics_records = []

    for f in npy_files:
        data = np.load(f, allow_pickle=True).item()
        m_name = data["model_name"]
        
        if "dense" in data and data["dense"].get("med") is not None:
            plot_data_dense[m_name] = {
                "med": data["dense"]["med"], 
                "L": data["dense"]["L"], 
                "U": data["dense"]["U"]
            }
            metrics_records.append({
                "Target": "Dense", 
                "Model": m_name,
                "MAE_raw": data["dense"]["MAE_raw"], 
                "MAE_log": data["dense"]["MAE_log"],
                "Tail": data["dense"]["Tail"], 
                "F1_Score": data["dense"]["F1"]
            })
            
        if "sparse" in data and data["sparse"].get("med") is not None:
            plot_data_sparse[m_name] = {
                "med": data["sparse"]["med"], 
                "L": data["sparse"]["L"], 
                "U": data["sparse"]["U"]
            }
            metrics_records.append({
                "Target": "Sparse", 
                "Model": m_name,
                "MAE_raw": data["sparse"]["MAE_raw"], 
                "MAE_log": data["sparse"]["MAE_log"],
                "Tail": data["sparse"]["Tail"], 
                "F1_Score": data["sparse"]["F1"]
            })

    df_metrics = pd.DataFrame(metrics_records)
    if not df_metrics.empty:
        df_metrics = df_metrics.sort_values(by=["Target", "Model"]).reset_index(drop=True)
        
    return plot_data_dense, plot_data_sparse, df_metrics

def plot_combined_predictive_bands(plot_data_dense, plot_data_sparse, y_true_dense, y_true_sparse, shift_dense=0.0, shift_sparse=0.0):
    model_sequence = ["True_Model", "AR2_GLM", "Full_GLM", "TFT_GLM", "TSM_GLM", "Two-Step", "TSM_Shrinkage"]
    proper_names = {
        "True_Model": "True DGP",
        "AR2_GLM": "AR2 Baseline",
        "Full_GLM": "Full GLM",
        "TFT_GLM": "TFT Bayesian head",
        "TSM_GLM": "TSMixer Bayesian head",
        "Two-Step": "Two-Step Shrinkage",
        "TSM_Shrinkage": "TSMixer + Shrinkage"
    }
    
    colors = ['#000000', '#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b'] 
    
    valid_seq = [m for m in model_sequence if (m in plot_data_dense or m in plot_data_sparse)]
    
    if not valid_seq:
        print("No valid models to plot.")
        return

    max_y_dense = max(np.max(y_true_dense), max([np.max(plot_data_dense[m]["U"] + shift_dense) for m in valid_seq if m in plot_data_dense]))
    max_y_sparse = max(np.max(y_true_sparse), max([np.max(plot_data_sparse[m]["U"] + shift_sparse) for m in valid_seq if m in plot_data_sparse]))
    
    ylim_dense = (0, max_y_dense * 1.05)
    ylim_sparse = (0, max_y_sparse * 1.05)
    
    fig, axes = plt.subplots(len(valid_seq), 2, figsize=(16, 2.8 * len(valid_seq)), sharex=True)
    
    if len(valid_seq) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle("95% Predictive Intervals and Anomaly Detection", fontsize=18, y=0.98, fontweight='bold')
    time_steps = np.arange(len(y_true_dense))
    
    for row_idx, model_key in enumerate(valid_seq):
        color = colors[model_sequence.index(model_key)]
        
        targets_info = [
            ("Dense", plot_data_dense, y_true_dense, shift_dense, ylim_dense),
            ("Sparse", plot_data_sparse, y_true_sparse, shift_sparse, ylim_sparse)
        ]
        
        for col_idx, (target_name, plot_data, y_true, current_shift, ylim) in enumerate(targets_info):
            ax = axes[row_idx, col_idx]
            
            if model_key not in plot_data:
                ax.text(0.5, 0.5, 'Not Evaluated', horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, fontsize=14, color='gray', style='italic')
                ax.set_title(f"{proper_names[model_key]} | Data Missing", loc='center', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            data = plot_data[model_key]
            shifted_U = data["U"] + current_shift
            original_tail = np.abs(0.025 - (np.sum(y_true > data["U"]) / len(y_true)))
            
            out_mask = y_true > shifted_U
            in_mask = ~out_mask
            
            ax.fill_between(time_steps, 0, shifted_U, color=color, alpha=0.3)
            ax.plot(time_steps, data["med"], color=color, linewidth=2)
            
            ax.plot(time_steps[in_mask], y_true[in_mask], 'k.', markersize=8)  
            ax.plot(time_steps[out_mask], y_true[out_mask], 'r.', markersize=10)
            
            sub_title = f"{proper_names[model_key]} | Tail Calibration (T): {original_tail:.3f}"
            if row_idx == 0:
                ax.set_title(f"{target_name} Target\n{sub_title}", loc='center', fontsize=14, fontweight='bold')
            else:
                ax.set_title(sub_title, loc='center', fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel("Counts", fontsize=11)
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(ylim)

    axes[-1, 0].set_xlabel("Test Set Time Steps", fontsize=12)
    axes[-1, 1].set_xlabel("Test Set Time Steps", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("combined_predictive_bands_adjusted.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    ds_base = SimDataset(lookback=10, horizon=1, step=1, flatten=False)
    y_all = ds_base.data.numpy()
    y_true_dense = y_all[TRAIN_LEN:, 0]
    y_true_sparse = y_all[TRAIN_LEN:, 1]

    plot_data_dense, plot_data_sparse, df_metrics = load_experiment_results(OUTPUT_DIR)
    
    print("\n=== Combined Metrics Table ===")
    if not df_metrics.empty:
        print(df_metrics.to_string())
    else:
        print("No metrics to display.")
        
    plot_combined_predictive_bands(
        plot_data_dense=plot_data_dense, 
        plot_data_sparse=plot_data_sparse, 
        y_true_dense=y_true_dense, 
        y_true_sparse=y_true_sparse
    )