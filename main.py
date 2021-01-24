import math
import os

import seaborn as sns
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from image_regression_dataset import ImageRegressionDataset
from model import MRIRegressor

# ! wget https://www.doc.ic.ac.uk/~bglocker/teaching/notebooks/brainage-data.zip
# ! unzip brainage-data.zip

if __name__ == "__main__":

    print("CUDA available: ", torch.cuda.is_available())
    sns.set(style="darkgrid")

    cuda_dev = "0"
    use_cuda = torch.cuda.is_available()

    data_dir = os.path.join("data", "brain_age")
    train_size = 0.5
    random_state = 42
    smoothen = 8
    edgen = False

    feats = 5
    num_epochs = 200
    lr = 0.006882801723742766
    gamma = 0.97958263796472
    batch_size = 4  # 128  # 32
    dropout_p = 0.5
    weight_decay = 0.005

    writer = SummaryWriter()

    device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
    print(f"Device: {device}", flush=True)
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(int(cuda_dev))}", flush=True)

    load_data = True
    if load_data:
        num_dl_workers = 0
        dataset = ImageRegressionDataset(data_dir, "train", smoothen, edgen, 3,
                                         load_data)
    else:
        num_dl_workers = 3
        dataset = ImageRegressionDataset(data_dir, "train", smoothen, edgen, 0,
                                         load_data)

    split_idx = math.floor(len(dataset) / 2)
    split = (split_idx, len(dataset) - split_idx)
    gen = torch.Generator().manual_seed(random_state)
    split1, split2 = torch.utils.data.random_split(dataset, split, gen)

    loss_function = L1Loss(reduction="none")

    params = sum(p.numel() for p in MRIRegressor(feats, dropout_p).parameters()
                 if p.requires_grad)
    print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, "
          f"Num_epochs: {num_epochs}, Gamma: {gamma}")
    print(f"Total Params: {params}")


    def eval_step(model, data_loader, loss_func, device):
        model.eval()
        with torch.no_grad():
            batch_losses = []
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                predicted_targets = model(data)
                losses = loss_func(predicted_targets, targets)
                batch_losses.append(losses.cpu())
        batch_losses = torch.cat(batch_losses)
        return batch_losses


    def train_step(model, train_data_loader, loss_func, optimizer, device):
        model.train()
        batch_losses = []
        for data, targets in train_data_loader:
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            predicted_targets = model(data)
            losses = loss_func(predicted_targets, targets)
            losses.sum().backward()
            optimizer.step()
            batch_losses.append(losses.detach().cpu())
        batch_losses = torch.cat(batch_losses)
        return batch_losses


    def log(main_tag, losses, epoch):
        writer.add_scalar(f"{main_tag}/max_loss", losses.max(), epoch)
        writer.add_scalar(f"{main_tag}/mean_loss", losses.mean(), epoch)
        writer.add_scalar(f"{main_tag}/std_loss", losses.std(), epoch)
        writer.add_scalar(f"{main_tag}/sum_loss", losses.sum(), epoch)


    dl_kwargs = dict(shuffle=True, num_workers=num_dl_workers)
    train_dataset, val_dataset = split1, split2
    for fold_id in range(2):
        model = MRIRegressor(feats, dropout_p).to(device=device)
        optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        train_dl = DataLoader(train_dataset, batch_size, **dl_kwargs)
        val_dl = DataLoader(val_dataset, batch_size, **dl_kwargs)

        epochs_iter = tqdm(range(num_epochs))
        epochs_iter.set_description(f"CV: {fold_id}")
        for epoch in epochs_iter:
            train_losses = train_step(model, train_dl, loss_function, optimizer,
                                      device)
            val_losses = eval_step(model, val_dl, loss_function, device)
            scheduler.step()
            log(f"{fold_id}/train", train_losses, epoch)
            log(f"{fold_id}/val", val_losses, epoch)
            writer.add_histogram(f"{fold_id}/train", train_losses, epoch)
            writer.add_histogram(f"{fold_id}/val", val_losses, epoch)

        train_dataset, val_dataset = val_dataset, train_dataset
