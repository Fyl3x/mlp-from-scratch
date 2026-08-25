import numpy as np

class ReLU():
    def __init__(self):
        pass
    def forward(self,x):
        out = np.maximum(x,0)
        return out
    def backward(self):
        pass