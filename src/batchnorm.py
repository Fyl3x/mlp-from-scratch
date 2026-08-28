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
        self.x = x
        self.loc_mean = x.mean(axis=0,keepdims=True)
        self.loc_var = x.var(axis=0,keepdims=True)
        self.x_norm = (x - self.loc_mean)/ np.sqrt(self.loc_var + self.eps)
        result = self.gamma * self.x_norm + self.beta
        return result 
    def backward(self,local_grad):
        self.grad = local_grad
        dresult = self.grad
        dx_norm = (dresult * self.gamma)
        dgamma = (dresult * self.x_norm).sum(axis=0)
        dbeta = dresult.sum(axis=0)


        dloc_var = (dx_norm * (-(1/2)*(self.x-self.loc_mean)*(self.loc_var+self.eps)**(-3/2))).sum(axis=0,keepdims=True)
        dloc_mean = (-dx_norm * (1/np.sqrt(self.loc_var + self.eps))).sum(axis=0,keepdims=True) #- dloc_var * (2/self.x.shape[0]) * (self.x - self.loc_mean) 
        dx = dx_norm * ((1/np.sqrt(self.loc_var + self.eps))) + dloc_var * (2/self.x.shape[0])*(self.x-self.loc_mean) + dloc_mean * (1/self.x.shape[0])
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx
    def parameters(self):
        params = super().parameters()
        params.append(self.gamma)
        params.append(self.beta)
        return params
    
    def gradients(self):
        result = super().gradients()
        result.append(self.dgamma)
        result.append(self.dbeta)
        return result
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    bn = BatchNorm1d(num_features=3)

    X = np.array([
        [1.0,  2.0,  3.0],
        [2.0,  4.0,  5.0],
        [3.0,  6.0,  7.0],
        [4.0,  8.0,  9.0]
    ])

    dout = np.random.randn(4, 3)

    # Forward + analytical backward
    bn(X)
    dX, dgamma, dbeta = bn.backward(dout)

    epsilon = 1e-5

    # -------------------------
    # Numerical dX
    # -------------------------
    numerical_dX = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X_plus = X.copy()
            X_plus[i, j] += epsilon

            X_minus = X.copy()
            X_minus[i, j] -= epsilon

            loss_plus = (BatchNorm1d(3)(X_plus) * dout).sum()
            loss_minus = (BatchNorm1d(3)(X_minus) * dout).sum()

            numerical_dX[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # -------------------------
    # Numerical dgamma
    # -------------------------
    numerical_dgamma = np.zeros(3)

    for j in range(3):

        gamma_plus = bn.gamma.copy()
        gamma_minus = bn.gamma.copy()

        gamma_plus[j] += epsilon
        gamma_minus[j] -= epsilon

        bn_plus = BatchNorm1d(3)
        bn_plus.gamma = gamma_plus

        bn_minus = BatchNorm1d(3)
        bn_minus.gamma = gamma_minus

        loss_plus = (bn_plus(X) * dout).sum()
        loss_minus = (bn_minus(X) * dout).sum()

        numerical_dgamma[j] = (loss_plus - loss_minus) / (2 * epsilon)

    # -------------------------
    # Numerical dbeta
    # -------------------------
    numerical_dbeta = np.zeros(3)

    for j in range(3):

        beta_plus = bn.beta.copy()
        beta_minus = bn.beta.copy()

        beta_plus[j] += epsilon
        beta_minus[j] -= epsilon

        bn_plus = BatchNorm1d(3)
        bn_plus.beta = beta_plus

        bn_minus = BatchNorm1d(3)
        bn_minus.beta = beta_minus

        loss_plus = (bn_plus(X) * dout).sum()
        loss_minus = (bn_minus(X) * dout).sum()

        numerical_dbeta[j] = (loss_plus - loss_minus) / (2 * epsilon)

    print("dX analytical:")
    print(dX)

    print("\ndX numerical:")
    print(numerical_dX)

    print("\ndX max difference:")
    print(np.max(np.abs(dX - numerical_dX)))

    print("\n\ndgamma analytical:")
    print(dgamma)

    print("\ndgamma numerical:")
    print(numerical_dgamma)

    print("\ndgamma max difference:")
    print(np.max(np.abs(dgamma - numerical_dgamma)))

    print("\n\ndbeta analytical:")
    print(dbeta)

    print("\ndbeta numerical:")
    print(numerical_dbeta)

    print("\ndbeta max difference:")
    print(np.max(np.abs(dbeta - numerical_dbeta)))