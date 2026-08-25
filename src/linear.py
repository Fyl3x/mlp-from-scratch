import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self,in_dim,out_dim,bias=True):
        self.in_dim = in_dim
        self.has_bias = bias
        self.out_dim = out_dim
        self.weight = np.random.randn(self.in_dim,self.out_dim) * np.sqrt(2/(self.in_dim))
        if bias == True:
            self.bias = np.zeros((1,self.out_dim)) 
    def forward(self,batch):
        out = batch @ self.weight # this should be (B,out_dim)
        if self.has_bias:
            out += self.bias
        return out
    def backward(self):
        pass

if __name__ == "__main__":

    lin = Linear(in_dim=64,out_dim=32)
    
    pooled = np.random.randn(5,64)
    out = lin(pooled)
    print(pooled.shape)
    print(out.shape)