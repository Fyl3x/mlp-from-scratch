from .module import Module
import cupy as np 


class Dropout(Module):
    def __init__(self,percentage=0.5):
        super().__init__()
        self.p = percentage

        self.training = True
        
    def forward(self,x):
        result = x
        if self.training:
            s1,s2 = x.shape
            holder_arr = np.random.rand(s1,s2)
            self.mask = holder_arr > self.p
            result = (x * self.mask) / (1-self.p)
        return result


    def backward(self,upstream_grad):
        result = upstream_grad
        if self.training:
            result = upstream_grad * self.mask/(1-self.p)
        return result
        
    def parameters(self):
        return super().parameters()
    def gradients(self):
        return super().gradients()