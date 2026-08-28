from .module import Module
class Optimizer(Module):
    def __init__(self,params,lr):
        self.params = params
        self.lr = lr
    def step(self,grads):
        for p,g in zip(self.params,grads):
            p -= self.lr*g
        
