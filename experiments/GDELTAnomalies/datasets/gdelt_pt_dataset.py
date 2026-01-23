import torch as pt
import pandas as pd
import numpy as np

class GDELTDataset(pt.utils.data.Dataset):
    def __init__(self, csv_location, lookback = 10, horizon = 1, step = 1):
        # Read data
        table = pd.read_csv(csv_location, index_col=0)
        self.weeks = table.index.to_numpy()
        self.columns = list(table.columns)
        self.data = pt.tensor(table.reset_index(drop=True).to_numpy(dtype="float32"), dtype=pt.float32)
        
        # Set parameters in case user wants to check them
        self.lookback = lookback
        self.horizon = horizon
        self.step = step

        # Calculate data partitions
        self.ts_partitions = pt.arange(self.data.shape[0]).unfold(0, lookback + horizon, step)

    def __len__(self):
        return len(self.ts_partitions)

    def __getitem__(self, index):
        # Get indices for this partition
        idxs = self.ts_partitions[index]
        
        # Get data for this partition
        data = self.data[idxs]
        X = data[:-1]
        y = data[-1]
        
        return X, y
