import torch as pt
import pandas as pd
import numpy as np
from pathlib import Path

class GDELTDataset(pt.utils.data.Dataset):
    def __init__(self, csv_location = "gdelt.csv", lookback = 10, horizon = 1, step = 1):
        # Read data
        directory = Path(__file__).parent.resolve()
        table = pd.read_csv(directory / csv_location, index_col=0)
        self.weeks = table.index.to_numpy()
        self.columns = list(table.columns)
        self.data = pt.tensor(table.reset_index(drop=True).to_numpy(dtype="float32"), dtype=pt.float32)

        # Static variables for this dataset
        self.country, self.event, self.lonAvg, self.latAvg = zip(*[x.split("_") for x in self.columns])
        
        # Longitude/Latitude encodings
        self.lonAvg = pt.tensor([int(x) for x in self.lonAvg], dtype=pt.float32)
        self.latAvg = pt.tensor([int(x) for x in self.latAvg], dtype=pt.float32)
        self.lonSin = pt.sin(self.lonAvg * pt.pi / 180)
        self.lonCos = pt.cos(self.lonAvg * pt.pi / 180)
        self.latSin = pt.sin(self.latAvg * pt.pi / 180)
        self.latCos = pt.cos(self.lonAvg * pt.pi / 180)

        # Country/Event encodings
        self.countryDf = pd.get_dummies(self.country)
        self.country = pt.tensor(self.countryDf.to_numpy(dtype="float32"))
        self.eventDf = pd.get_dummies(self.event)
        self.event = pt.tensor(self.eventDf.to_numpy(dtype="float32"))

        # Set parameters in case user wants to check them
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.num_series = self.event.shape[0]

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

if __name__ == "__main__":
    data = GDELTDataset()