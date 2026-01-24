import torch as pt
import numpy as np
import tqdm
from omegaconf import OmegaConf

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
import GDELTAnomalies.models.tft as tft

def load_dataset(rank, world_size):
    dataset = GDELTDataset(lookback=10, horizon=1, step=1, flatten=True, dtype=pt.float16)

    data_len = len(dataset)
    train_len = 308 * dataset.num_series
    valid_len = 52 * dataset.num_series


    train_data = pt.utils.data.Subset(dataset, range(train_len))
    valid_data = pt.utils.data.Subset(dataset, range(train_len, train_len + valid_len))
    # test_data = pt.utils.data.Subset(dataset, range(train_len + valid_len, data_len))

    train_sampler = pt.utils.data.DistributedSampler(train_data, world_size, rank, True)
    valid_sampler = pt.utils.data.DistributedSampler(valid_data, world_size, rank, False)


    train_dataloader = pt.utils.data.DataLoader(train_data, batch_size=1024 * 4, sampler=train_sampler, num_workers=3, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    valid_dataloader = pt.utils.data.DataLoader(valid_data, batch_size=1024 * 4, sampler=valid_sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    # test_dataloader = pt.utils.data.DataLoader(test_data, batch_size=1024 * 8, shuffle=False, num_workers=2, pin_memory=True)
    
    
    return train_dataloader, valid_dataloader, train_sampler, valid_sampler

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
    
    train_dataloader, valid_dataloader, train_sampler, valid_sampler = load_dataset(rank, world_size)

    device = pt.device("cuda")
    pt.manual_seed(854923)
    scaler = pt.amp.GradScaler()

    data_props = {
        'num_historical_numeric': 4200,
        'num_historical_categorical': 0,
        'num_static_numeric': 26,
        'num_static_categorical': 0,
        'num_future_numeric': 0,
        'num_future_categorical': 0,
        "num_future_steps": 1,
        "device": "cuda"
    }

    configuration = {
        'model': {
                'dropout': 0.05,
                'state_size': 4,
                'output_quantiles': [0.5],
                'lstm_layers': 2,
                'attention_heads': 2
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
    if rank == 0:
        tqdm_iter = tqdm.tqdm(range(epochs))

    for epoch in range(epochs):
        # Init samplers
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

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
        model.join()
        valid_history[epoch] = valid_loss
        if rank == 0:
            tqdm_iter.set_postfix_str(f"{valid_loss=}")
        
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
            
        if epoch % 100 == 0:
            scheduler.step()

        if rank == 0 and (epoch % 10 == 0):
            pt.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "valid_loss": valid_loss,
                "valid_history": pt.tensor(valid_history),
            }, f"checkpoints/TFT_{epoch}.pt")

        if rank == 0:
            tqdm_iter.update(1)

    cleanup()

def setup():
    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = pt.accelerator.current_accelerator()
    backend = pt.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    pt.distributed.init_process_group(backend, device_id=0)
    return pt.distributed.get_rank(), pt.distributed.get_world_size()
    

def cleanup():
    pt.distributed.destroy_process_group()

if __name__ == "__main__":
    train()