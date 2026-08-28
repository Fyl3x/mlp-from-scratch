from .linear import Linear
from .relu import ReLU
from .module import Module
from .batchnorm import BatchNorm1d
from .crossentropy import CrossEntropyLoss
class MLPHead(Module):
    def __init__(self,embed_dim,output_dim,targets):
        super().__init__()
        self.targets = targets

        input_dim = embed_dim
        out_dim = output_dim
        self.linear1 = Linear(input_dim,4*input_dim,bias=False)
        self.ln1 = BatchNorm1d(4*input_dim)
        self.relu1 = ReLU()
        self.linear2 = Linear(4*input_dim,input_dim,bias=False)
        self.ln2 = BatchNorm1d(input_dim)
        self.relu2 = ReLU()
        self.linear3 = Linear(input_dim,out_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        self.x = x 
        return x 
    def parameters(self):
        result = super().parameters()
        result = result + (self.linear1.parameters())
        result = result + (self.ln1.parameters())
        result = result + (self.relu1.parameters())
        result = result + (self.linear2.parameters())
        result = result + (self.ln2.parameters())
        result = result + (self.linear3.parameters())
        return result
    def gradients(self):
        result = super().gradients()
        result = result + (self.linear1.gradients())
        result = result + (self.ln1.gradients())
        result = result + (self.relu1.gradients())
        result = result + (self.linear2.gradients())
        result = result + (self.ln2.gradients())
        result = result + (self.linear3.gradients())
        return result
    def backward(self):
        loss = CrossEntropyLoss(self.x,self.targets)
        losser = loss()
        dlogits = loss.backward()
        g = self.linear3.backward(dlogits)
        g = self.relu2.backward(g)
        g = self.ln2.backward(g)
        g = self.linear2.backward(g)
        g = self.relu1.backward(g)
        g = self.ln1.backward(g)
        g = self.linear1.backward(g)
        return g 
                        

