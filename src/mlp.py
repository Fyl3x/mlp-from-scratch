from .linear import Linear
from .relu import ReLU
from .module import Module
class MLPHead(Module):
    def __init__(self,embed_dim,output_dim):
        input_dim = embed_dim
        out_dim = output_dim
        self.linear1 = Linear(input_dim,4*input_dim)
        self.relu1 = ReLU()
        self.linear2 = Linear(4*input_dim,input_dim)
        self.relu2 = ReLU()
        self.linear3 = Linear(input_dim,out_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x 
    
                        

