import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_experiment_folder(root="."):
    experiment_id = 0
    while True:
        fn = os.path.join(root, f'Subjects/Subject_{experiment_id}')
        if not os.path.exists(fn):
            print(f"Making {experiment_id}")
            os.makedirs(fn)
            with open(f'{fn}/log.txt', 'w+') as log:
                log.write('\n')
            break
        else:
            print(f"Subject_{experiment_id} exists")
            experiment_id += 1
    return fn


def train_epoch(model, dl_loader, loss_function, optimizer, scheduler, device):
    model.train()
    batch_losses = []
    for batch_data, batch_labels in dl_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        batch_predictions = model(batch_data)
        loss = loss_function(batch_predictions, batch_labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    scheduler.step()
    return batch_losses


def eval_epoch(model, dl_loader, loss_function, device):
    model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch_data, batch_labels in dl_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            batch_losses.append(loss.item())
    return batch_losses


def plot_loss(x_epochs_mean_losses, y_epochs_mean_losses, labels, fn):
    num_epochs = len(x_epochs_mean_losses)
    plt.plot([epoch for epoch in range(num_epochs)], x_epochs_mean_losses, color='b', label=labels[0])
    plt.plot(np.linspace(start=0, stop=num_epochs-1, num=len(y_epochs_mean_losses)),
             y_epochs_mean_losses, color='r', label=labels[1])
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 30)
    plt.xlim(-5, num_epochs + 5)
    plt.legend()
    plt.savefig(fn)
    plt.close()


def plot_cv_losses(folds_training_losses, folds_val_losses, num_epochs, fn):
    for j in range(2):
        plt.plot([epoch for epoch in range(num_epochs)], folds_training_losses[j], color='b', label=f'Train-{j}')
        plt.plot([5 * i for i in range(len(folds_val_losses[j]))], folds_val_losses[j], color='r', label=f'Val-{j}')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 40)
    plt.xlim(-5, num_epochs + 5)
    plt.legend()
    plt.savefig(fn)
    plt.close()


def log_results(cv_mean_age_error, num_epochs, batch_size,
                lr, feats, gamma, smoothen,
                edgen, dropout_p, params, model, fn):
    result = f"""
    ########################################################################
    # Score = {cv_mean_age_error}

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

    with open(fn, 'a+') as log:
        log.write('\n')
        log.write(result)
        log.write('\n')
