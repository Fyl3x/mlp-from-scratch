import numpy as np
from embedding import emb,batch
class Linear():
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = np.random.randn(self.in_dim,self.out_dim) * np.sqrt(2/(self.in_dim))
        self.bias = np.zeros((1,self.out_dim)) 
    def forward(self,batch):
        out = batch @ self.weight # this should be (B,out_dim)
        out += self.bias
        return out
    def backward(self):
        pass



lin = Linear(in_dim=64,out_dim=32)

pooled = emb.forward(batch)
out = lin.forward(pooled)
print(pooled.shape)
print(out.shape)