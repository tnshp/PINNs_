import torch
from model import DNN
import pandas as pd
import numpy as np


DEVICE = 'cuda'
torch.set_default_dtype(torch.float32)

data = pd.read_csv('heat_equation\DATA\sparse_data.csv', header=None)
#data = np.reshape(data, (len(data), 50,50))

x_train = data.iloc[:,:3].to_numpy()
y_train = data.iloc[:,3].to_numpy()

x_train = torch.tensor(x_train, requires_grad=True).to(DEVICE)
y_train = torch.tensor(y_train, requires_grad=True).to(DEVICE)


#Continous Model parameters 
layers = [3,128,128,128,128,128,128,128,1]
alpha  = 2
ivp = 0
model  = DNN(layers,alpha=alpha, ivp =  ivp)



