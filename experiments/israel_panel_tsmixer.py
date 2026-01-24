import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
from GDELTAnomalies.models.tsmixer import TSMixer

import optuna

print(pt.accelerator.current_accelerator())


def objective(trial: optuna.Trial):
    lookback = trial.suggest_int("lookback", 5, 52)
    
    dataset = GDELTDataset(lookback=10, horizon=1, step=1)

    data_len = len(dataset)
    train_len = 308
    valid_len = 52


    train_data = pt.utils.data.Subset(dataset, range(train_len))
    valid_data = pt.utils.data.Subset(dataset, range(train_len, train_len + valid_len))
    test_data = pt.utils.data.Subset(dataset, range(train_len + valid_len, data_len))

    train_dataloader = pt.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)
    valid_dataloader = pt.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_dataloader = pt.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)