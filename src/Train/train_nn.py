import torch
import torchinfo as ti
from torch import nn
from torch.nn import Module, Linear, Sequential, MSELoss, Dropout
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from time import time
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from src.Preprocessing import *
from config import *


class TGVDataset(Dataset):
    """
    Custom dataset for the TGV data

    Args:
    -----
        `filename` (str): path to the csv file containing the data
        `train` (bool): whether to return the training or testing set

    Attributes:
    -----------
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
        `is_train` (bool): whether to return the training or testing set

    Methods:
    --------
        `__getitem__(self, i)`: returns the i-th element of the dataset
        `__len__(self)`: returns the length of the dataset
    """

    def __init__(self, filename, train=True):
        super().__init__()
        self.X_train, self.y_train, self.X_test, self.y_test = data_pipeline_1(
            filename, scaler="minmax")
        self.X_train, self.y_train, self.X_test, self.y_test = self.X_train.values, self.y_train.values, self.X_test.values, self.y_test.values
        self.X_train, self.y_train, self.X_test, self.y_test = torch.Tensor(self.X_train), torch.Tensor(
            self.y_train.reshape(-1, 1)), torch.Tensor(self.X_test), torch.Tensor(self.y_test.reshape(-1, 1))
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
    """
    Dense neural network

    Args:
    -----
        `input_size` (int): size of the input layer
        `*hidden_sizes` (int): sizes of the hidden layers

    Attributes:
    -----------
        `sizes` (list): sizes of the layers
        `sequs` (list): list of the layers of the network

    Methods:
    --------
        `forward(self, X)`: forward pass of the network
        `info(self)`: prints the summary of the network
    """

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


def get_model(dataset):
    """
    Returns the model to use for the given dataset

    Args:
    -----
        `dataset` (Dataset): dataset to use

    Returns:
    --------
        `model` (Module): model to use
    """
    x, y = dataset[0]
    model = DenseNN(x.shape[0], 128, 64)
    return model


def train(model: Module, dataloaders, optimizer, lr, loss_fn, scoring_fn, n_epochs, step_cp=30, epoch_cp=5):
    """
    Trains the model

    Args:
    -----
        `model` (Module): model to train
        `dataloaders` (tuple): tuple of the training and testing dataloaders
        `optimizer` (Optimizer): optimizer to use
        `lr` (float): learning rate
        `loss_fn` (Loss): loss function to use
        `scoring_fn` (function): scoring function to use
        `n_epochs` (int): number of epochs to train the model
        `step_cp` (int): number of batches between each checkpoint
        `epoch_cp` (int): number of epochs between each checkpoint

    Returns:
    --------
        `model` (Module): trained model
    """
    device = torch.device("cpu")
    train_data, eval_data = dataloaders
    optimizer = optimizer(model.parameters(), lr=lr)

    def epoch_train(epoch):
        """
        Trains the model on the training set for one epoch

        Args:
        -----
            `epoch` (int): current epoch

        Returns:
        --------
            `total_loss` (float): average loss on the training set
            `total_score` (float): average score on the training set
        """
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
        """
        Evaluates the model on the testing set

        Returns:
        --------
            `total_loss` (float): average loss on the testing set
            `total_score` (float): average score on the testing set
        """
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
    """
    Plots the predictions of the model on the testing set

    Args:
    -----
        `model` (Module): trained model
        `eval_data` (DataLoader): testing set
        `scoring_fn` (function): scoring function to use
    """
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
          MSELoss(), mean_squared_error, 1000, 50, epoch_cp=50)

    plot_test(model, dataloaders[1], mean_squared_error)


if __name__ == "__main__":
    main()
