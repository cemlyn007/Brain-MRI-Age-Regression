from torch.nn import L1Loss
from torch.optim import Adam, lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from image_regression_dataset import ImageRegressionDataset
from model import MRIRegressor
from experiment_functions import train_epoch, eval_epoch, get_experiment_folder, plot_loss, log_results, plot_cv_losses

# ! wget https://www.doc.ic.ac.uk/~bglocker/teaching/notebooks/brainage-data.zip
# ! unzip brainage-data.zip

print("CUDA available: ", torch.cuda.is_available())
sns.set(style='darkgrid')

"""# Parameters"""
data_dir = 'data/brain_age/'
training_size = 0.5
random_state = 42
smoothen = 8
edgen = False
train_val_meta_data_csv_path = os.path.join(data_dir, 'meta', 'meta_data_reg_train.csv')
test_meta_data_csv_path = os.path.join(data_dir, 'meta', 'meta_data_reg_test.csv')

cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
num_workers = max(os.cpu_count() - 2, 0)

dtype = torch.float32
feats = 5
num_epochs = 200
lr = 0.006882801723742766
gamma = 0.97958263796472
batch_size = 32
dropout_p = 0.5
weight_decay = 0.005

"""# Device Setup"""
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

"""# Dataset & Preprocessing"""
meta_data_reg_train = pd.read_csv(train_val_meta_data_csv_path)
ids = meta_data_reg_train['subject_id'].tolist()
ages = meta_data_reg_train['age'].tolist()

"""# Train/Val Dataset Initialisation"""
X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(ids, ages, train_size=training_size, random_state=random_state)
dataset1 = ImageRegressionDataset(data_dir, X_fold1, y_fold1, smoothen, edgen)
dataset2 = ImageRegressionDataset(data_dir, X_fold2, y_fold2, smoothen, edgen)

print("Creating Subject Folder")
fn = get_experiment_folder(".")
print("Created Subject Folder")

loss_function = L1Loss()

params = sum(p.numel() for p in MRIRegressor(feats, dropout_p).parameters() if p.requires_grad)
print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")
print(f"Total Params: {params}")

train_loader = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers)
val_loader = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers)
folds_training_losses = []
folds_val_mean_losses = []

for i in range(2):
    print("CV: ", i)
    train_epochs_mean_losses = []
    val_epochs_mean_losses = []
    val_epochs = []
    i_fold_val_scores = []

    model = MRIRegressor(feats, dropout_p).to(device=device)
    optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    for epoch in range(num_epochs):
        epoch_train_batch_losses = train_epoch(model, train_loader, loss_function, optimizer, scheduler, device)
        train_mean_loss = np.mean(epoch_train_batch_losses)
        train_max_loss = np.max(epoch_train_batch_losses)
        train_epochs_mean_losses.append(train_mean_loss)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_epochs.append(epoch)
            epoch_val_batch_losses = eval_epoch(model, val_loader, loss_function, device)
            val_mean_loss = np.mean(epoch_val_batch_losses)
            val_epochs_mean_losses.append(val_mean_loss)
            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            print(f"Epoch: {epoch} "
                  f"Train Maxiumum Batch Loss: {train_max_loss} "
                  f"Train Mean Batch Loss: {train_mean_loss}, "
                  f"Val Mean Batch Loss: {val_mean_loss}")
        train_loader, val_loader = val_loader, train_loader

    i_fold_val_losses = eval_epoch(model, val_loader, loss_function, device)
    i_fold_mean_fold_loss = np.mean(i_fold_val_losses)
    folds_val_mean_losses.append(i_fold_mean_fold_loss)
    print(f"Fold {i}, Validation Mean Loss: {i_fold_mean_fold_loss}")
    plot_loss(train_epochs_mean_losses, val_epochs, val_epochs_mean_losses, ["Train", "Val"], f'{fn}/graph_{i}.png')
    folds_training_losses.append(train_epochs_mean_losses)
    torch.save(model, f'{fn}/model-{i}.pth')

cv_mean_age_error = np.mean(folds_val_mean_losses)
print(f"CV Mean Loss on whole val set: {cv_mean_age_error}")

log_results(cv_mean_age_error, num_epochs, batch_size,
            lr, feats, gamma, smoothen,
            edgen, dropout_p, params, model, f'{fn}/log.txt')

plot_cv_losses(folds_training_losses, val_epochs, folds_val_mean_losses, num_epochs, f'{fn}/cv_graph.png')

"""# Full Train & Final Test"""

meta_data_reg_test = pd.read_csv(test_meta_data_csv_path)
test_ids = meta_data_reg_test['subject_id'].tolist()
test_ages = meta_data_reg_test['age'].tolist()

train_ds = ImageRegressionDataset(data_dir, ids, ages, smoothen, edgen)
test_ds = ImageRegressionDataset(data_dir, test_ids, test_ages, smoothen, edgen)

loss_function = L1Loss()
train_loader = DataLoader(train_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

train_epochs_mean_losses = []
test_epochs = []
test_epochs_mean_losses = []

model = MRIRegressor(feats, dropout_p).to(device=device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {params}")

model = MRIRegressor(feats, dropout_p).to(device=device)
optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

for epoch in range(num_epochs):
    epoch_train_batch_losses = train_epoch(model, train_loader, loss_function, optimizer, scheduler, device)
    train_mean_loss = np.mean(epoch_train_batch_losses)
    train_max_loss = np.max(epoch_train_batch_losses)
    train_epochs_mean_losses.append(train_mean_loss)
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        test_epochs.append(epoch)
        epoch_test_batch_losses = eval_epoch(model, val_loader, loss_function, device)
        test_mean_loss = np.mean(epoch_test_batch_losses)
        test_epochs_mean_losses.append(test_mean_loss)
        print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
        print(f"Epoch: {epoch} "
              f"Train Maxiumum Batch Loss: {train_max_loss} "
              f"Train Mean Batch Loss: {train_mean_loss}, "
              f"Test Mean Batch Loss: {test_mean_loss}")

test_losses = eval_epoch(model, test_loader, loss_function, device)
test_mean_loss = np.mean(test_losses)
print(f"Test Mean Batch Loss: {test_mean_loss}")

plot_loss(train_epochs_mean_losses, test_epochs, test_epochs_mean_losses, ["Train", "Test"],
          f'{fn}/test_loss_graph.png')
torch.save(model, f'{fn}/final_model.pth')

model.eval()
train_pred_ages = []
train_actual_ages = []
test_pred_ages = []
test_actual_ages = []

with torch.no_grad():
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device=device)
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        train_pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
        train_actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        test_pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
        test_actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

pred_ages = np.concatenate(train_pred_ages + test_pred_ages)
actual_ages = np.concatenate(train_actual_ages + test_actual_ages)

fig, ax = plt.subplots()
ax.scatter(actual_ages, pred_ages, marker='.')
ax.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'k--', lw=2)
ax.set_xlabel('Real Age')
ax.set_ylabel('Predicted Age')
plt.savefig(f'{fn}/age_vs_predict.png')
plt.close()

print("Train Mean Age Error: ", np.mean(np.abs(np.array(train_pred_ages) - np.array(train_actual_ages))))
print("Test Mean Age Error: ", np.mean(np.abs(np.array(test_pred_ages) - np.array(test_actual_ages))))
