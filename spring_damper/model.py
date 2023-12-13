import torch 
import torch.nn as nn
import torch.autograd as autograd
from collections import OrderedDict


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
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
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
class CustomLoss(nn.Module):
    def __init__(self,mass, k, b):
        super(CustomLoss, self).__init__()
        self.mass = mass
        self.k = k
        self.b = b

    def loss_f(self,x, t):
        x_t = autograd.grad(x, t, grad_outputs = torch.ones_like((x)),
                retain_graph = True, create_graph = True)[0]

        x_t_t = autograd.grad(x_t, t, grad_outputs = torch.ones_like((x)),
                retain_graph = True, create_graph = True)[0]
        f =  self.mass * x_t_t + self.b * x_t + self.k * x
        return f
    
    def loss_ivp():
        pass
    
    def forward(self, t, x,  x_targets):
        loss_mse = torch.square(torch.sub(x , x_targets))
        loss = torch.mean(loss_mse.mean()) + torch.mean(self.loss_f(x,t) ** 2)
        return loss