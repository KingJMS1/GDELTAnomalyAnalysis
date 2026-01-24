import torch as pt
import numpy as np
import tqdm
from omegaconf import OmegaConf

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
import GDELTAnomalies.models.tft as tft

print(pt.accelerator.current_accelerator())

dataset = GDELTDataset(lookback=10, horizon=1, step=1, flatten=True, dtype=pt.float16)

data_len = len(dataset)
train_len = 308 * dataset.num_series
valid_len = 52 * dataset.num_series


train_data = pt.utils.data.Subset(dataset, range(train_len))
valid_data = pt.utils.data.Subset(dataset, range(train_len, train_len + valid_len))
test_data = pt.utils.data.Subset(dataset, range(train_len + valid_len, data_len))

train_dataloader = pt.utils.data.DataLoader(dataset, batch_size=1024 * 16, shuffle=True, num_workers=3, pin_memory=True, prefetch_factor=4, persistent_workers=True)
valid_dataloader = pt.utils.data.DataLoader(dataset, batch_size=1024 * 16, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
test_dataloader = pt.utils.data.DataLoader(dataset, batch_size=1024 * 16, shuffle=False, num_workers=2, pin_memory=True)

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

device = pt.device("cuda")
pt.manual_seed(854923)


data_props = {'num_historical_numeric': 4200,
                'num_historical_categorical': 0,
                'num_static_numeric': 26,
                'num_static_categorical': 0,
                'num_future_numeric': 0,
                'num_future_categorical': 0,
                "num_future_steps": 1,
                "device": "cuda"
                # 'historical_categorical_cardinalities': [],
                # 'static_categorical_cardinalities': [],
                # 'future_categorical_cardinalities': [],
                }

configuration = {
    'model':
        {
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

model = tft.TemporalFusionTransformer(OmegaConf.create(configuration)).to(device).to(pt.float16)
optimizer = pt.optim.Adam(model.parameters(), lr=4e-5)
scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

quantile = 0.5
lossfn = lambda x, y: quantile_loss(x, y, quantile)


valid_history = np.zeros(epochs)

tqdm_iter = tqdm.tqdm(range(epochs))

for epoch in tqdm_iter:
    # Calculate validation loss
    model.eval()
    valid_loss = 0
    with pt.no_grad():
        for X, y, static in valid_dataloader:
            X = X.to(device)
            y = y.to(device)
            static = static.to(device)

            batch = {
                "historical_ts_numeric": X,
                "static_feats_numeric": static,
            }

            stuff = model.forward(batch)
            pred = stuff["predicted_quantiles"]
            valid_loss += lossfn(pred.squeeze(), pt.log(y + 1))
    valid_history[epoch] = valid_loss
    tqdm_iter.set_postfix_str(f"{valid_loss=}")
    
    model.train()
    for X, y, static in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        static = static.to(device)
        batch = {
            "historical_ts_numeric": X,
            "static_feats_numeric": static,
        }
        
        stuff = model.forward(batch)
        pred = stuff["predicted_quantiles"]
        loss = lossfn(pred.squeeze(), pt.log(y + 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        scheduler.step()

pt.save({
    "epoch": 3000,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "valid_loss": valid_loss,
    "valid_history": pt.tensor(valid_history),
}, "checkpoints/TFT_3000.pt")