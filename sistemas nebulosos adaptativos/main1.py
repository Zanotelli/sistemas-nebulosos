import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import math
import skfuzzy
import pandas as pd
import random as rand
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
import sys
sys.path.append("./anfis-pytorch/")
from anfispytorchmaster.experimental import train_anfis, test_anfis, plot_all_mfs
from anfispytorchmaster.membership import GaussMembFunc, make_gauss_mfs
import anfispytorchmaster.anfis as ANFIS

iris = open('Iris.csv', mode='r')
breast = open('Breast.csv', mode='r')

#################################################################################


# loading data set
data = loadmat('mg.mat')
data = pd.DataFrame(data['x'])
sgn = data.values.reshape(-1,)
len(sgn)



#train test
size = len(sgn)
train_s = int(size * 0.8)
test_s = size - train_s
x_train = sgn[:train_s]
x_test = sgn[train_s:]



def model_serie():
    invardefs = [
            ('xm18', make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('xm12', make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('xm6',  make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('x',    make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ]
    outvars = ['xp6']
    model = ANFIS.AnfisNet('model_define', invardefs, outvars)
    return model


def our_data(data):
    num_cases = len(data) - 18
    x = torch.zeros((num_cases, 4))
    y = torch.zeros((num_cases, 1))
    for t in range(18, len(data)-6):
            values = [data[t-18],data[t-12],data[t-6],data[t],data[t+6]]
            x[t-18] = torch.tensor(values[0:4])
            y[t-18] = values[4]
    dl = DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=True)
    return dl


model = model_serie()
train_data = our_data(x_train)
train_anfis(model, train_data, 100, True)
test_data = our_data(x_test)
test_anfis(model, test_data, True)

