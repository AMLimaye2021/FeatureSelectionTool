import sys
import numpy as np
from matplotlib import pyplot
import pandas as pd
import h5py
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import GEOparse

class PrimarySiteDataset(Dataset):
    def __init__(self, expression, classifications, batch_size=None):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss

        self.length = expression.shape[0]


        self.x_data = torch.from_numpy(expression).type(x_dtype)
        self.y_data = torch.from_numpy(classifications).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

print("USING pytorch VERSION: ", torch.__version__)
# data1 = GEOparse.get_GEO(filepath="./GDS2947.soft.gz") # Adenoma/Healthy Set (Testing?) (True Count) Samples 64
data2 = GEOparse.get_GEO(filepath="./GDS4379.soft.gz") # Diseased Carcinoma set (Transformed Count) Samples 62
# data3 = GEOparse.get_GEO(filepath="./GDS4381.soft.gz") # Tumor Set (Transformed Count) Samples 64
# data4 = GEOparse.get_GEO(filepath="./GDS4393.soft.gz") # Metastatic/Tumor Set (True Count) Samples 54
# data5 = GEOparse.get_GEO(filepath="./GDS4513.soft.gz") # Tumor/Excised Set (Transformed count) Samples 53




def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # convert 1-hot vectors back into indices
    max_values, target_index = y.max(dim=1)
    target_index = target_index.type(torch.LongTensor)

    max_predict_values, predict_index = y_predict.max(dim=1)
    predict_index = predict_index.type(torch.LongTensor)

    loss = loss_fn(y_predict, target_index)
    #loss = loss_fn(predict_index, target_index)

    # Compute loss.
    #loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item()

def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return losses

def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict

def test(model, loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y, y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)
    y_true_vector = np.concatenate(y_vectors)

    return y_predict_vector, y_true_vector


def accuracy(y_predict, y_vector):
    y_predict_argmax = np.zeros(len(y_predict), dtype=int)
    y_vector_argmax = np.zeros(len(y_vector), dtype=int)

    for i in range(len(y_predict)):
        y_predict_argmax[i] = y_predict[i].argmax()
        y_vector_argmax[i] = y_vector[i].argmax()

    print(y_predict_argmax, y_vector_argmax)
    # print(y_predict_argmax[0], y_vector_argmax[0])
    match = 0
    for j in range(len(y_predict)):
        if y_predict_argmax[j] == y_vector_argmax[j]:
            match += 1
    print(match / len(y_predict_argmax) * 100, '%')
    return y_predict_argmax, y_vector_argmax

class ShallowLinear(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''

    def __init__(self):
        # Perform initialization of the pytorch superclass
        super(ShallowLinear, self).__init__()

        # Define network layer dimensions
        D_in, H1, H2, H3, H4, D_out = [54675, 5000, 100, 2500, 50, 2]  # These numbers correspond to each layer: [input, hidden_1, output]

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, H4)
        self.linear5 = nn.Linear(H4, D_out)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.linear1(x)  # hidden layer
        x = torch.relu(x)  # activation function
        x = self.linear2(x)  # output layer
        x = torch.relu(x)  # activation function
        x = self.linear3(x)  # hidden layer
        x = torch.relu(x)  # activation function
        x = self.linear4(x)  # hidden layer
        x = torch.relu(x)  # activation function
        x = self.linear5(x)
        # x = F.log_softmax(x) # softmax to convert to probability

        return x

def run(dataset_train, dataset_test):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 4

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    # Define the hyperparameters
    learning_rate = 1e-4
    # learning_rate = 1e-2
    shallow_model = ShallowLinear()

    n_epochs = 4

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(shallow_model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()  # Cross Entropy Loss (CEL)
    # loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    loss = train(model=shallow_model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn,
                 epochs=n_epochs)

    # Test and get the resulting predicted y values
    y_predict, y_vector = test(model=shallow_model, loader=data_loader_test)

    return loss, y_predict, y_vector

def getValues(data):
    table = data.table
    table_expression = np.array(table)
    table_info = np.array(data.columns)
    table_info = table_info[:, 1]
    gene_table = table_expression[:, 1]
    table_expression = table_expression[:, 2:]
    table_expression = table_expression.astype(np.float_)
    table_expression = table_expression.transpose()
    return table_expression, table_info

def main():
    data6 = GEOparse.get_GEO(filepath="./GDS4516.soft.gz")# Metastatic/Stage 3 Set (Transformed count) Samples 104
    data2 = GEOparse.get_GEO(filepath="./GDS4379.soft.gz")
    table6 = data6.table
    table6_expression = np.array(table6)
    table6_info = np.array(data6.columns)
    table6_info = table6_info[:, 1]
    gene_table = table6_expression[:, 1]
    table6_expression = table6_expression[:, 2:]
    table6_expression = table6_expression.astype(np.float_)
    table6_expression = table6_expression.transpose()
    table6_expression2, table6_info2 = getValues(data6)
    infolen = table6_info.shape
    infolen = infolen[0]
    classification = np.zeros((infolen, 2), dtype=int)
    for i in range(len(table6_info)):
        if (table6_info[i] == "no metastasis"):
            classification[i][0] = 1
        else:
            classification[i][1] = 1

    dataset_train = PrimarySiteDataset(table6_expression, classification)
    dataset_test = PrimarySiteDataset(table6_expression, classification)
    l = len(dataset_train)
    print(dataset_train[0])

    print("Train set size: ", dataset_train.length)

    losses, y_predict, y_vector = run(dataset_train=dataset_train, dataset_test=dataset_test)

    print(y_predict.shape, y_vector.shape)
    print('y_predict     y')
    for i in range(len(y_predict[0])):
        print('{:0.3f}    '.format(y_predict[0][i]), y_vector[0][i])

    y_predict_argmax, y_vector_argmax = accuracy(y_predict=y_predict, y_vector=y_vector)
    print(y_predict_argmax.shape, y_vector_argmax.shape)
    print(y_predict_argmax[0], y_vector_argmax[0])


if __name__ == "__main__":
    main()
