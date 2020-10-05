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
num_workers = os.cpu_count() - 2

dtype = torch.float32
feats = 5
num_epochs = 200
lr = 0.006882801723742766
gamma = 0.97958263796472
batch_size = 32
dropout_p = 0.5

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

"""# Model Architecture"""

print("Creating Subject Folder")
experiment_id = 0
while True:
    fn = f'Subjects/Subject_{experiment_id}'
    if not os.path.exists(fn):
        print(f"Making {experiment_id}")
        os.makedirs(fn)
        with open(f'{fn}/log.txt', 'w+') as log:
            log.write('\n')
        break
    else:
        print(f"Subject_{experiment_id} exists")
        experiment_id += 1
print("Created Subject Folder")

loss_function = L1Loss()

params = sum(p.numel() for p in MRIRegressor(feats, dropout_p).parameters() if p.requires_grad)
print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")
print(f"Total Params: {params}")

train_loader = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers)
val_loader = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers)
folds_val_scores = []
folds_training_losses = []
folds_val_losses = []

for i in range(2):
    print("CV: ", i)
    training_loss = []
    val_loss_epochs = []
    i_fold_val_scores = []

    model = MRIRegressor(feats, dropout_p).to(device=device)
    optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.to(device=device)
            batch_data = batch_data.to(device=device)
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        training_MAE = np.mean(epoch_loss)
        training_loss.append(training_MAE)

        scheduler.step()

        if epoch % 5 == 0:
            val_loss = []
            model.eval()
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device=device)
                    batch_labels = batch_labels.to(device=device)
                    batch_preds = model(batch_data)
                    loss = loss_function(batch_preds, batch_labels)
                    val_loss.append(loss.item())
                mean_val_errors = np.mean(val_loss)
                val_loss_epochs.append(mean_val_errors)
            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            print(f"{experiment_id}::{i} "
                  f"Train Maxiumum Age Error: {np.round(np.max(epoch_loss))} "
                  f"Train Mean Age Error: {training_MAE}, "
                  f"Val Mean Age Error: {mean_val_errors}")

        train_loader, val_loader = val_loader, train_loader

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            i_fold_val_scores.append(loss.item())

    mean_fold_score = np.mean(i_fold_val_scores)
    val_loss_epochs.append(mean_fold_score)
    print(f"Mean Age Error: {mean_fold_score}")

    folds_val_scores.append(mean_fold_score)

    plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
    plt.plot([5 * i for i in range(len(val_loss_epochs))], val_loss_epochs, color='r', label='Val')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 30)
    plt.xlim(-5, num_epochs + 5)
    plt.legend()
    plt.savefig(f'{fn}/graph_{i}.png')
    plt.close()

    folds_training_losses.append(training_loss)
    folds_val_losses.append(val_loss_epochs)
    torch.save(model, f'{fn}/model-{i}.pth')

final_MAE = np.mean(folds_val_scores)
print(f"Average Loss on whole val set: {final_MAE}")

result = f"""
########################################################################
# Score = {final_MAE}

# Number of epochs:
num_epochs = {num_epochs}

# Batch size during training
batch_size = {batch_size}

# Learning rate for optimizers
lr = {lr}

# Size of feature amplifier
Feature Amplifier: {feats}


# Gamma (using sched)
Gamma: {gamma}

# Smooth:
smoothen = {smoothen}

# Edgen:
edgen = {edgen}

# Amount of dropout:
dropout_p = {dropout_p}

Total number of parameters is: {params}

# Model:
{str(model)}
########################################################################
"""

with open(f'{fn}/log.txt', 'a+') as log:
    log.write('\n')
    log.write(result)
    log.write('\n')

for j in range(i):
    plt.plot([epoch for epoch in range(num_epochs)], folds_training_losses[j], color='b', label=f'Train-{j}')
    plt.plot([5 * i for i in range(len(folds_val_losses[j]))], folds_val_losses[j], color='r', label=f'Val-{j}')
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.ylim(0, 40)
plt.xlim(-5, num_epochs + 5)
plt.legend()
plt.savefig(f'{fn}/cv_graph.png')
plt.close()

"""# Full Train & Final Test"""

meta_data_reg_test = pd.read_csv(test_meta_data_csv_path)
test_ids = meta_data_reg_test['subject_id'].tolist()
test_ages = meta_data_reg_test['age'].tolist()

train_ds = ImageRegressionDataset(data_dir, ids, ages, smoothen, edgen)
test_ds = ImageRegressionDataset(data_dir, test_ids, test_ages, smoothen, edgen)

loss_function = L1Loss()
train_loader = DataLoader(train_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

training_loss = []
test_loss_epochs = []

model = MRIRegressor(feats, dropout_p).to(device=device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {params}")

optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = []
    for batch_data, batch_labels in train_loader:
        batch_labels = batch_labels.to(device=device)
        batch_data = batch_data.to(device=device)
        batch_preds = model(batch_data)
        loss = loss_function(batch_preds, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    training_MAE = np.mean(epoch_loss)
    training_loss.append(training_MAE)

    scheduler.step()

    if epoch % 5 == 0:
        test_loss = []
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device=device)
                batch_labels = batch_labels.to(device=device)
                batch_preds = model(batch_data)
                loss = loss_function(batch_preds, batch_labels)
                test_loss.append(loss.item())
            mean_test_errors = np.mean(test_loss)
            test_loss_epochs.append(mean_test_errors)
        print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
        print(f"{experiment_id}:: "
              f"Train Maxiumum Age Error: {np.round(np.max(epoch_loss))} "
              f"Train Mean Age Error: {training_MAE} "
              f"Test Mean Age Error: {mean_test_errors}")

model.eval()
test_scores = []
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device=device)
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        loss = loss_function(batch_preds, batch_labels)
        test_scores.append(loss.item())

score = np.mean(test_scores)
test_loss_epochs.append(score)
print(f"Mean Age Error: {score}")

plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
plt.plot([5 * i for i in range(len(test_loss_epochs))], test_loss_epochs, color='r', label='Test')
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.ylim(0, 30)
plt.xlim(-5, num_epochs + 5)
plt.legend()
plt.savefig(f'{fn}/test_loss_graph.png')
plt.close()
print(f"Average Loss on whole test set: {score}")

torch.save(model, f'{fn}/final_model.pth')

model.eval()
pred_ages = []
actual_ages = []
with torch.no_grad():
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
        actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
        actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

pred_ages = np.concatenate(pred_ages)
actual_ages = np.concatenate(actual_ages)

y = actual_ages
predicted = pred_ages

fig, ax = plt.subplots()
ax.scatter(y, predicted, marker='.')
ax.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
ax.set_xlabel('Real Age')
ax.set_ylabel('Predicted Age')
plt.savefig(f'{fn}/scatter_part_c.png')
plt.close()
