import torch as pt

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import itertools as it


# PyTorch time series dataset for this data
class SimData(pt.utils.data.Dataset):
    def __init__(self, seed=1207, num_signal = 6, num_noise = 500, simlen = 400, mc_n = 5000, noise_level = 0.4, lookback = 10):
        # Simulate the dataset
        np.random.seed(seed)
        serieslabels = np.ones(num_noise + num_signal, dtype="bool") * np.False_
        signalIdxs = np.random.choice(np.arange(num_noise + num_signal), num_signal, replace=False)
        serieslabels[signalIdxs] = True

        def apply_signals(data, signals, j):
            newdata = np.zeros_like(data)
            for i, signal in enumerate(signals):
                newdata[i] = signal(data, j)
            return newdata

        # Simulation signals, and their means
        signals = [
            lambda x, j: x[0] + np.random.normal(0, noise_level) + 0.5 * np.sin(j / 2),
            lambda x, j: np.exp(0.2 * (np.random.normal(0, noise_level) + x[0])) + 0.1 * (x[1] + 0.5 * np.sin(j / 2)),
            lambda x, j: x[0] + x[1] + np.random.normal(0, noise_level),
            lambda x, j: np.random.exponential(1) * (x[2] + np.random.normal(0, noise_level) + 0.5 * np.sin(j / 4)),
            lambda x, j: x[4] + np.random.normal(0, noise_level) + 0.5 * np.sin(j),
            lambda x, j: 2 * np.sin((x[4] + x[0]) / 2) + np.random.normal(0, noise_level) * 0.5
        ]
        def signal_mean(data, j, mc_n=mc_n):
            newdata = np.zeros_like(data)
            newdata[0] = data[0] + 0.5 * np.sin(j / 2)
            newdata[1] = np.mean(np.exp(0.2 * np.random.normal(0, noise_level, mc_n) + data[0]) + 0.1 * (data[1] + 0.5 * np.sin(j / 2)))
            newdata[2] = data[0] + data[1]
            newdata[3] = np.mean(np.random.exponential(1, mc_n) * (data[2] + np.random.normal(0, noise_level, mc_n) + 0.5 * np.sin(j / 4)))
            newdata[4] = data[4] + 0.5 * np.sin(j)
            newdata[5] = 2 * np.sin((data[4] + data[0]) / 2)

            return newdata


        # Setup output datasets
        data = np.zeros((simlen, num_signal + num_noise))
        oracle = np.zeros((simlen, num_signal + num_noise))

        # Simulate dataset
        data[0, ~serieslabels] = np.random.normal(0, 2, num_noise)
        data[0, serieslabels] = apply_signals(np.zeros(num_signal), signals, 0)
        periods = np.random.exponential(2, num_noise)
        for i in range(1, simlen):
            data[i, ~serieslabels] = data[i - 1, ~serieslabels] + np.random.normal(0, noise_level, num_noise) + 0.5 * np.sin(i / periods)
            data[i, serieslabels] = apply_signals(data[i - 1, serieslabels], signals, i)
            oracle[i, ~serieslabels] = data[i - 1, ~serieslabels] + 0.5 * np.sin(i / periods)
            oracle[i, serieslabels] = signal_mean(data[i - 1, serieslabels], i)
        
        # Final outputs
        self.countData = np.round((data / 0.5) + 1) * (data > 0)
        self.countOracle = np.round((oracle / 0.5) + 1) * (oracle > 0)
        self.sparsity = np.mean(self.countData == 0, axis=0)
        self.lookback = lookback
        self.datIdxs = np.lib.stride_tricks.sliding_window_view(np.arange(self.countData.shape[0]), lookback + 1)
        self.num_series = num_noise + num_signal
        self.signal_indicator = serieslabels


    def __len__(self):
        return len(self.datIdxs)

    def __getitem__(self, index):
        data = pt.tensor(self.countData[self.datIdxs[index]], dtype=pt.float32)
        return data[:self.lookback], data[-1]