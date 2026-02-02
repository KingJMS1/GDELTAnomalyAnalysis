import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import optuna

from GDELTAnomalies.datasets.gdelt_pt_dataset import GDELTDataset
from GDELTAnomalies.models.tsmixer import TSMixer

import gc




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


def objective(trial: optuna.Trial):
    gc.collect()
    pt.cuda.empty_cache()
    
    lookback = trial.suggest_int("lookback", 5, 52)
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    ff_dim = trial.suggest_int("ff_dim", 8, 128)

    dataset = GDELTDataset(lookback=lookback, horizon=1, step=1, event_filter = ["04", "13", "18", "19"])

    data_len = len(dataset)
    train_len = 256
    valid_len = 52


    train_data = pt.utils.data.Subset(dataset, range(train_len))
    valid_data = pt.utils.data.Subset(dataset, range(train_len, train_len + valid_len))
    # test_data = pt.utils.data.Subset(dataset, range(train_len + valid_len, data_len))

    train_dataloader = pt.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)
    valid_dataloader = pt.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    # test_dataloader = pt.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = pt.device("cuda")
    pt.manual_seed(854923)

    model = TSMixer(lookback, 1, dataset.num_series, num_blocks=num_blocks, ff_dim=ff_dim).to(device)

    quantile = 0.5
    lossfn = lambda x, y: quantile_loss(x, y, quantile)

    epochs = 8000
    tqdm_iter = tqdm.tqdm(range(epochs))

    optimizer = pt.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

    valid_history = np.zeros(epochs)

    for epoch in tqdm_iter:
        # Calculate validation loss
        model.eval()
        valid_loss = 0
        with pt.no_grad():
            for X, y in valid_dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = model.forward(X)
                valid_loss += lossfn(pred.squeeze(), pt.log(y + 1))
        valid_history[epoch] = valid_loss
        tqdm_iter.set_postfix_str(f"{valid_loss=}")
        
        model.train()
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            
            pred = model.forward(X)
            loss = lossfn(pred.squeeze(), pt.log(y + 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            scheduler.step()
        
    pt.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "valid_loss": valid_loss
    }, f"checkpoints/TSMixer_ukr_small_{ff_dim=}_{lookback=}_{num_blocks=}.pt")

    return valid_loss

if __name__ == "__main__":
    print(pt.accelerator.current_accelerator())
    storage_name = f"sqlite:///studies/TSMixer_ukr_small.db"
    #no-name-16720e9a-65e4-4bc8-92db-19b812c1f122
    study = optuna.create_study(study_name="TSMixer_ukr_small", storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=20)