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
        self.batch = batch
        out = batch @ self.weight # this should be (B,out_dim)
        if self.has_bias:
            out += self.bias
        return out
    def backward(self,local_grad):
        self.local_grad = local_grad
        dout = local_grad
        self.dW = self.batch.transpose() @ dout
        dX = dout @ self.weight.transpose()
        if self.has_bias:
            self.db = dout.sum(axis=0,keepdims=True)
        return dX
    def gradients(self):
        result  = super().gradients()
        # result = []
        result.append(self.dW)
        if self.has_bias == True:
            result.append(self.db)
        return result
    def parameters(self):

        params = super().parameters()
        
        params.append(self.weight)
        if self.has_bias == True:
            params.append(self.bias)
        return params


# if __name__ == "__main__":
#     import numpy as np

#     np.random.seed(42)

#     lin = Linear(in_dim=4, out_dim=3, bias=True)

#     X = np.random.randn(2, 4)
#     dout = np.random.randn(2, 3)

#     out = lin(X)
#     dX, dW, db = lin.backward(dout)

#     epsilon = 1e-5

#     # Check dX[0,0]
#     X_plus = X.copy()
#     X_plus[0,0] += epsilon

#     X_minus = X.copy()
#     X_minus[0,0] -= epsilon

#     loss_plus = (lin(X_plus) * dout).sum()
#     loss_minus = (lin(X_minus) * dout).sum()

#     numerical_dX = (loss_plus - loss_minus) / (2 * epsilon)

#     print("dX analytical:", dX[0,0])
#     print("dX numerical: ", numerical_dX)
#     print("difference:   ", abs(dX[0,0] - numerical_dX))
#     # Check dW[0,0]

#     W_plus = lin.weight.copy()
#     W_plus[0,0] += epsilon

#     W_minus = lin.weight.copy()
#     W_minus[0,0] -= epsilon

#     lin.weight = W_plus
#     loss_plus = (lin(X) * dout).sum()

#     lin.weight = W_minus
#     loss_minus = (lin(X) * dout).sum()

#     lin.weight = W_plus  # restore

#     numerical_dW = (loss_plus - loss_minus) / (2 * epsilon)

#     print("dW analytical:", dW[0,0])
#     print("dW numerical: ", numerical_dW)
#     print("difference:   ", abs(dW[0,0] - numerical_dW))
#     # Check db[0,0]

#     bias_plus = lin.bias.copy()
#     bias_plus[0,0] += epsilon

#     bias_minus = lin.bias.copy()
#     bias_minus[0,0] -= epsilon

#     lin.bias = bias_plus
#     loss_plus = (lin(X) * dout).sum()

#     lin.bias = bias_minus
#     loss_minus = (lin(X) * dout).sum()

#     lin.bias = bias_plus  # restore

#     numerical_db = (loss_plus - loss_minus) / (2 * epsilon)

#     print("db analytical:", db[0,0])
#     print("db numerical: ", numerical_db)
#     print("difference:   ", abs(db[0,0] - numerical_db))