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

def gera_dados(dados):
    n = len(dados) - 18
    x = torch.zeros((n, 4))
    y = torch.zeros((n, 1))
    for t in range(18, len(dados)-6):
            aux_values = [dados[t-18], dados[t-12], dados[t-6], dados[t], dados[t+6]]
            x[t-18] = torch.tensor(aux_values[0:4])
            y[t-18] = aux_values[4]
    dl = DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=True)
    return dl

def inicializa_problema():
    x1 = [
            ('t-12', make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('t-12', make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('t-6',  make_gauss_mfs(0.1, [0.425606, 1.313696])),
            ('t',    make_gauss_mfs(0.1, [0.425606, 1.313696]))]
    x2 = ['t+6']
    return ANFIS.AnfisNet('Previsão de uma Séria Temporal', x1, x2)


dados = np.ravel(loadmat('mg.mat')['x'])
N = len(dados)
dados_train = dados[0:int(N*.8)]
dados_te = dados[int(N*.8)+1:N]

modelo = inicializa_problema()
dados_treino_modelados = gera_dados(dados_train)
train_anfis(modelo, dados_treino_modelados, 100, True)

dados_teste_modelados = gera_dados(dados_te)
test_anfis(modelo, dados_teste_modelados, True)