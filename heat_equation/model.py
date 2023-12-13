import torch 
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from collections import OrderedDict


class DNN(torch.nn.Module):
    def __init__(self, layers, alpha, ivp, optimizer = torch.optim.adam):
        super(DNN, self).__init__()

        self.optimizer = optimizer
        # parameters
        self.alpha = alpha
        self.ivp = ivp

        self.top = np.linspace([0,0,0], [93,50,0], 10) 
        #self.bottom =  

        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        self.layers = nn.Sequential(layerDict)
        
    def forward(self, inputs):    #inputs (t,x,y)
        out = self.layers(inputs) 
        return out    
    
    def loss_(self, inputs, u_targets):

        u = self.forward(inputs)

        loss_mse = torch.square( u - u_targets)

        u_x = autograd.grad(u, inputs[:,1], grad_outputs = torch.ones_like((inputs[:,1])),
                retain_graph = True, create_graph = True)[0]

        u_x_x = autograd.grad(u_x, inputs[:,1], grad_outputs = torch.ones_like((u_x)),
                retain_graph = True, create_graph = True)[0]
        
        u_y = autograd.grad(u, inputs[:,2], grad_outputs = torch.ones_like((inputs[:,2])),
                retain_graph = True, create_graph = True)[0]

        u_y_y = autograd.grad(u_y, inputs[:,2], grad_outputs = torch.ones_like((u_y)),
                retain_graph = True, create_graph = True)[0]
        
        u_t = autograd.grad(u, inputs[:,0], grad_outputs = torch.ones_like((inputs[:,0])),
                retain_graph = True, create_graph = True)[0]
        
        loss_f = (self.alpha ** 2 ) * (u_x_x + u_y_y) - u_t

        #loss_ivp =  (self.ivp -   self.forward()) ** 2

        #loss_bvp  =  

        loss = torch.mean(loss_mse) + torch.mean(loss_f ** 2) 
        return loss
    
    def train(self, inputs, u_targets, epochs = 500):
         for epoch in range(epochs):
            loss = self.loss_(inputs, u_targets)
            print("Epoch: ", epoch, "Loss: ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
