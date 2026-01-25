import torch as pt
import pandas as pd
import numpy as np
from pathlib import Path

class GDELTDataset(pt.utils.data.Dataset):
    """
    GDELT Dataset. Format (of y):
            Series 1    Series 2    ...
    Week 1  val         val
    Week 2  val         val
    ...
    
    The lonAvg, latAvg, lonSin, lonCos, latSin, latCos tell which column is at which location
    country and event tell which column is which country or event
    
    If flatten = True, dataset columns are flattened such that the format for y is now:
    Week 1 Series 1, Week 1 Series 2, ..., Week 2 Series 1, Week 2 Series 2, ...
    """
    def __init__(self, csv_location = "gdelt.csv", lookback = 10, horizon = 1, step = 1, flatten = False, dtype=pt.float32, return_index = False):
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
        self.flatten = flatten
        self.dtype = dtype
        self.return_index = return_index

        # Calculate data partitions
        self.ts_partitions = pt.arange(self.data.shape[0]).unfold(0, lookback + horizon, step)
        self.num_times = len(self.ts_partitions)

    def __len__(self):
        if self.flatten:
            return self.num_times * self.num_series
        else:
            return self.num_times

    def __getitem__(self, index):
        if self.flatten:
            # Figure out which time and series we are supposed to be in
            series = index % self.num_series
            timeIdx = index // self.num_series
            times = self.ts_partitions[timeIdx]

            # Get our data
            data = self.data[times]
            X = data[:-1]
            y = data[-1, series]

            # Also return corresponding statics for y
            lonSin = self.lonSin[series]
            lonCos = self.lonCos[series]
            latSin = self.latSin[series]
            latCos = self.latCos[series]
            country = self.country[series]
            event = self.event[series]
            statics = pt.concat((pt.tensor([lonSin, lonCos, latSin, latCos]), country, event))

            if self.return_index:
                return X.to(self.dtype), y.to(self.dtype), statics.to(self.dtype), series, timeIdx
            return X.to(self.dtype), y.to(self.dtype), statics.to(self.dtype)
        else:
            # Get indices for this partition
            idxs = self.ts_partitions[index]
            
            # Get data for this partition
            data = self.data[idxs]
            X = data[:-1]
            y = data[-1]
            
            if self.return_index:
                return X.to(self.dtype), y.to(self.dtype), index
            
            return X.to(self.dtype), y.to(self.dtype)

if __name__ == "__main__":
    data = GDELTDataset()