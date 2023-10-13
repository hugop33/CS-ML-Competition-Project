import torch
import torchinfo as ti
from torch import nn
from torch.nn import Module, Linear, ReLU, Sequential, MSELoss, Dropout, Tanh
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from time import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from src.Preprocessing import *
from config import *


class TGVDataset(Dataset):
    def __init__(self, filename, train=True):
        super().__init__()
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(
            filename)
        self.X_train, self.y_train, self.X_test, self.y_test = torch.Tensor(self.X_train), torch.Tensor(
            self.y_train), torch.Tensor(self.X_test), torch.Tensor(self.y_test)
        self.is_train = train

    def __getitem__(self, i):
        if self.is_train:
            return self.X_train[i], self.y_train[i]

        else:
            return self.X_test[i], self.y_test[i]

    def __len__(self):
        if self.is_train:
            return len(self.X_train)
        else:
            return len(self.X_test)


class DenseNN(Module):
    def __init__(self, input_size, *hidden_sizes):
        super().__init__()
        self.sizes = [input_size, *hidden_sizes]
        self.sequs = [Sequential(Linear(self.sizes[i], self.sizes[i+1]),
                                 Dropout(0.5)) for i in range(len(self.sizes)-1)]
        self.sequs.append(Sequential(Linear(self.sizes[-1], 1)))
        self.sequs = nn.ModuleList(self.sequs)

    def forward(self, X):
        for i in range(len(self.sequs)):
            X = self.sequs[i](X)
        return X

    def info(self):
        ti.summary(self, (1, self.sizes[0]), device='cpu')


def scale(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    # Xscaler = MinMaxScaler()
    # yscaler = MinMaxScaler()
    Xscaler = StandardScaler()
    yscaler = StandardScaler()
    X_train = Xscaler.fit_transform(X_train)
    X_test = Xscaler.transform(X_test)
    y_train, y_test = y_train.values.reshape(-1,
                                             1), y_test.values.reshape(-1, 1)
    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)
    return X_train, y_train, X_test, y_test


def load_data(csv_name):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    df = pd.read_csv(csv_path, sep=';')
    df = date_to_float_col(df, replace=False)
    df["depart_region"] = df["gare_depart"].apply(gare_region)
    df["arrivee_region"] = df["gare_arrivee"].apply(gare_region)
    df["depart_departement"] = df["gare_depart"].apply(gare_departement)
    df["arrivee_departement"] = df["gare_arrivee"].apply(gare_departement)
    # Remove gare_arrivee, gare_depart
    df = df.drop(labels=["gare_arrivee", "gare_depart"], axis=1)
    # One hot encoding
    df = oneHotEncoding(df, "depart_region")
    df = oneHotEncoding(df, "arrivee_region")
    df = oneHotEncoding(df, "depart_departement")
    df = oneHotEncoding(df, "arrivee_departement")
    df = oneHotEncoding(df, "service")

    # df["duree_moyenne"] = df["duree_moyenne"]-df["retard_moyen_arrivee"]x

    xcols = ["duree_moyenne", "nb_train_prevu", "annee", "mois", "date"]

    xcols_to_keep = [c for c in df.columns if c in xcols or c.startswith(
        ("depart_region", "arrivee_region", "depart_departement", "arrivee_departement", "service"))]
    ycols = ["retard_moyen_arrivee"]

    df = df[xcols_to_keep+ycols]

    X_train, y_train, X_test, y_test = train_test(df)

    return scale(X_train, y_train, X_test, y_test)


def get_model(dataset):
    x, y = dataset[0]
    model = DenseNN(x.shape[0], 128, 512, 128)
    return model


def train(model: Module, dataloaders, optimizer, lr, loss_fn, scoring_fn, n_epochs, step_cp=30, epoch_cp=5):
    device = torch.device("cpu")
    train_data, eval_data = dataloaders
    optimizer = optimizer(model.parameters(), lr=lr)

    def epoch_train(epoch):
        model.train()
        model.to(device)
        total_loss = 0
        total_score = 0
        for i, (x, y) in enumerate(train_data):
            batch_start = time()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            out = outputs.detach().numpy()
            y = y.detach().numpy()
            score = scoring_fn(out, y)
            total_score += score
            if (i % step_cp == 0 or i == len(train_data) - 1) and epoch % epoch_cp == 0:
                elapsed = time()-batch_start
                remaining = (len(train_data)-1-i)*elapsed
                print(
                    f"\tBatch {i}"+" "*(len(str(len(train_data)))-len(str(i))) + f"\t- Loss: {round(loss.item(), 4)}\t- MSE: {round(score/len(x), 4)}\t- Time remaining in epoch: {int(remaining)}s")
        return total_loss/len(train_data), total_score/len(train_data)

    def epoch_test():
        model.eval()
        model.to(device)
        total_loss = 0
        total_score = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(eval_data):
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item()
                score = scoring_fn(outputs, y)
                total_score += score
        return total_loss/len(eval_data), total_score/len(eval_data)

    for epoch in range(n_epochs):
        if epoch % epoch_cp == 0:
            start = time()
            print(f"Epoch {epoch+1}")
        train_loss, train_score = epoch_train(epoch)
        test_loss, test_score = epoch_test()
        if epoch % epoch_cp == 0:
            print(f"\tTrain loss: {round(train_loss, 4)}\t- Train MSE: {round(train_score, 4)}\t- Test loss: {round(test_loss, 4)}\t- Test MSE: {round(test_score, 4)}\t- Time elapsed: {int(time()-start)}s")
            print("Time remaining:", int((n_epochs-epoch-1)*(time()-start)), "s")
    return model


def plot_test(model, eval_data, scoring_fn):
    device = torch.device("cpu")
    model.eval()
    model.to(device)
    y_pred = []
    y_test = []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_data):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            y_pred.extend(outputs.detach().numpy())
            y_test.extend(y.detach().numpy())
    print("Test Score :", scoring_fn(y_test, y_pred))

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Boston test and predicted data")
    plt.legend()
    plt.show()


def main():
    train_dataset, eval_dataset = TGVDataset(
        DATA_FILENAME), TGVDataset(DATA_FILENAME, train=False)
    dataloaders = (DataLoader(train_dataset, batch_size=128, shuffle=True),
                   DataLoader(eval_dataset, batch_size=128, shuffle=False))
    model = get_model(train_dataset)
    train(model, dataloaders, Adam, 1e-4,
          MSELoss(), mean_squared_error, 500, 50, epoch_cp=50)

    plot_test(model, dataloaders[1], mean_squared_error)


if __name__ == "__main__":
    main()
