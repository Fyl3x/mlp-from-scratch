from .module import Module
import numpy as np


class LayerNorm1d(Module):
    def __init__(self,num_features,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
    def forward(self,x):
        loc_mean = x.mean(axis=1,keepdims=True)
        loc_var = x.var(axis=1,keepdims=True)
        normalized_x = (x-loc_mean)/np.sqrt(loc_var + self.eps)
        result = normalized_x * self.gamma + self.beta
        return result
    def backward(self):
        pass