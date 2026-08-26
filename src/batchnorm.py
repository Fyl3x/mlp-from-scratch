from .module import Module
import numpy as np


class BatchNorm1d(Module):
    def __init__(self,num_features,eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((num_features,))
        self.beta =  np.zeros((num_features,))
    def forward(self,x):
        loc_mean = x.mean(axis=0,keepdims=True)
        loc_var = x.var(axis=0,keepdims=True)
        x_norm = (x - loc_mean)/ np.sqrt(loc_var + self.eps)
        result = self.gamma * x_norm + self.beta
        return result 
    def backward(self):
        pass