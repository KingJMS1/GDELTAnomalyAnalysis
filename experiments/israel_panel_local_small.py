import torch as pt
import numpy as np
import tqdm
from omegaconf import OmegaConf

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
import GDELTAnomalies.models.tft as tft

import sys

def load_dataset(rank, world_size, test = False):
    dataset = GDELTDataset(lookback=10, horizon=1, step=1, flatten=True, dtype=pt.float16, return_index=test, event_filter=["04", "13", "18", "19"])

    data_len = len(dataset)
    train_len = 308 * dataset.num_series
    valid_len = 52 * dataset.num_series

    test_data = None
    test_sampler = None
    test_dataloader = None

    train_data = pt.utils.data.Subset(dataset, range(train_len))
    valid_data = pt.utils.data.Subset(dataset, range(train_len, train_len + valid_len))
    if test:
        test_data = pt.utils.data.Subset(dataset, range(train_len + valid_len, data_len))

    # train_sampler = pt.utils.data.DistributedSampler(train_data, world_size, rank, True)
    # valid_sampler = pt.utils.data.DistributedSampler(valid_data, world_size, rank, False)
    # if test:
    #     test_sampler = pt.utils.data.DistributedSampler(test_data, world_size, rank, False)

    train_dataloader = pt.utils.data.DataLoader(train_data, batch_size=32, num_workers=3, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    valid_dataloader = pt.utils.data.DataLoader(valid_data, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True)
    if test:
        test_dataloader = pt.utils.data.DataLoader(test_data, batch_size=32, sampler=test_sampler, num_workers=2, pin_memory=True, persistent_workers=True)    
    
    if test:
        return train_dataloader, valid_dataloader, test_dataloader, None, None, None, dataset
    return train_dataloader, valid_dataloader, None, None, dataset

def quantile_loss(y_pred, y_true, q):
    """
    Calculate the quantile loss (pinball loss).

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        q (float or torch.Tensor): The quantile level (0 to 1).

    Returns:
        torch.Tensor: The mean quantile loss.
    """
    errors = y_true - y_pred
    loss = pt.max(q * errors, (q - 1) * errors)
    return pt.mean(loss)

def train():
    rank, world_size = setup()
    
    train_dataloader, valid_dataloader, _, _, dataset = load_dataset(rank, world_size)

    device = pt.device("cuda")
    pt.manual_seed(854923)
    scaler = pt.amp.GradScaler()

    data_props = {
        'num_historical_numeric': dataset.num_series,
        'num_historical_categorical': 0,
        'num_static_numeric': dataset.num_statics,
        'num_static_categorical': 0,
        'num_future_numeric': 0,
        'num_future_categorical': 0,
        "num_future_steps": 1,
        "device": "cuda"
    }

    configuration = {
        'model': {
                'dropout': 0.05,
                'state_size': 24,
                'output_quantiles': [0.5],
                'lstm_layers': 2,
                'attention_heads': 3
        },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    epochs = 3000

    tft_model = tft.TemporalFusionTransformer(OmegaConf.create(configuration)).to(device)
    model = pt.nn.parallel.DistributedDataParallel(tft_model,)
    optimizer = pt.optim.Adam(model.parameters(), lr=4e-5)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

    quantile = 0.5
    lossfn = lambda x, y: quantile_loss(x, y, quantile)


    valid_history = np.zeros(epochs)

    tqdm_iter = None

    for epoch in range(epochs):
        # Init samplers
        # train_sampler.set_epoch(epoch)
        # valid_sampler.set_epoch(epoch)

        if (epoch == 0) and (rank == 0):
            print("Starting validation")

        # Calculate validation loss
        model.eval()
        valid_loss = 0
        with pt.no_grad():
            for X, y, static in valid_dataloader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                static = static.to(device, non_blocking=True)

                batch = {
                    "historical_ts_numeric": X,
                    "static_feats_numeric": static,
                }

                with pt.autocast("cuda"):
                    stuff = model.forward(batch)
                    pred = stuff["predicted_quantiles"]
                    valid_loss += lossfn(pred.squeeze(), pt.log(y + 1))
        # model.join()
        valid_history[epoch] = valid_loss
        if rank == 0:
            print(f"{epoch=}, {valid_loss=}", flush=True)
            tqdm_iter = tqdm.tqdm(total=len(train_dataloader))

        model.train()
        for X, y, static in train_dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            batch = {
                "historical_ts_numeric": X,
                "static_feats_numeric": static,
            }
            
            loss = None
            with pt.autocast("cuda"):
                stuff = model.forward(batch)
                pred = stuff["predicted_quantiles"]
                loss = lossfn(pred.squeeze(), pt.log(y + 1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if rank == 0:
                tqdm_iter.update(1)
            
        if epoch % 100 == 0:
            scheduler.step()

        if rank == 0 and (epoch % 10 == 0):
            pt.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_history": pt.tensor(valid_history),
            }, f"checkpoints/TFT_isr_small/TFT_hdim_24_{epoch}.pt")


    cleanup()

def predict():
    rank, world_size = setup()
    
    train_dataloader, valid_dataloader, test_dataloader, _, _, _, dataset = load_dataset(rank, world_size, test=True)

    device = pt.device("cuda")
    pt.manual_seed(854923)

    data_props = {
        'num_historical_numeric': dataset.num_series,
        'num_historical_categorical': 0,
        'num_static_numeric': dataset.num_statics,
        'num_static_categorical': 0,
        'num_future_numeric': 0,
        'num_future_categorical': 0,
        "num_future_steps": 1,
        "device": "cuda"
    }

    configuration = {
        'model': {
                'dropout': 0.05,
                'state_size': 24,
                'output_quantiles': [0.5],
                'lstm_layers': 2,
                'attention_heads': 3
        },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    # Load our model
    checkpoint = pt.load("checkpoints/TFT_isr_small_hdim_24_1300.pt")
    model = tft.TemporalFusionTransformer(OmegaConf.create(configuration)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model = pt.nn.parallel.DistributedDataParallel(tft_model,)

    predictions = {"train": [], "train_indices": [], "validation": [], "validation_indices": [], "test": [], "test_indices": []}
    
    model.eval()

    with pt.no_grad():
        if rank == 0:
            print("Validation")
        
        for X, y, static, series, timeIdx in tqdm.tqdm(valid_dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)

            batch = {
                "historical_ts_numeric": X,
                "static_feats_numeric": static,
            }

            with pt.autocast("cuda"):
                stuff = model.forward(batch)
                predictions["validation"].append(stuff["predicted_quantiles"].detach().cpu())
                predictions["validation_indices"].append((series, timeIdx))

        if rank == 0:
            print("Test")
            i = 0

        for X, y, static, series, timeIdx in tqdm.tqdm(test_dataloader):
            if rank == 0:
                print(f"TBatch {i}")
                i += 1
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)

            batch = {
                "historical_ts_numeric": X,
                "static_feats_numeric": static,
            }

            with pt.autocast("cuda"):
                stuff = model.forward(batch)
                predictions["test"].append(stuff["predicted_quantiles"].detach().cpu())
                predictions["test_indices"].append((series, timeIdx))

        if rank == 0:
            print("Training")

        for X, y, static, series, timeIdx in tqdm.tqdm(train_dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)

            batch = {
                "historical_ts_numeric": X,
                "static_feats_numeric": static,
            }

            with pt.autocast("cuda"):
                stuff = model.forward(batch)
                predictions["train"].append(stuff["predicted_quantiles"].detach().cpu())
                predictions["train_indices"].append((series, timeIdx))

    pt.save(predictions, f"checkpoints/TFT_isr_small_preds.pt")
        

def setup():
    return 0, 1
    

def cleanup():
    pass

if __name__ == "__main__":
    predict()