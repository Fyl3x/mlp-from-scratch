from .module import Module
from .backend import np

class LayerNorm1d(Module):
    def __init__(self,num_features,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
    def forward(self,x):
        self.x = x 
        self.loc_mean = x.mean(axis=1,keepdims=True)
        self.loc_var = x.var(axis=1,keepdims=True)
        self.normalized_x = (x-self.loc_mean)/np.sqrt(self.loc_var + self.eps)
        result = self.normalized_x * self.gamma + self.beta
        return result
    def backward(self,local_grad):
        self.grad = local_grad
        dresult = self.grad 
        dnormalized_x = dresult * self.gamma
        dgamma = (dresult * self.normalized_x).sum(axis=0)
        dbeta = (dresult).sum(axis=0)
        dloc_mean = (-dnormalized_x *(1/(np.sqrt(self.loc_var + self.eps)))).sum(axis=1,keepdims=True)
        dloc_var = (dnormalized_x * (self.x - self.loc_mean) * (-1/2) * (self.loc_var + self.eps)**(-3/2)).sum(axis=1,keepdims=True)
        

        dx = dnormalized_x *(1/np.sqrt(self.loc_var + self.eps)) + dloc_var * (2/self.x.shape[1])*(self.x-self.loc_mean)  + dloc_mean * (1/self.x.shape[1]) 
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
        

    np.random.seed(42)

    B = 4
    C = 3

    X = np.random.randn(B, C)
    dout = np.random.randn(B, C)

    ln = LayerNorm1d(num_features=C)

    # Analytical gradients
    ln(X)
    dX = ln.backward(dout)
    dgamma, dbeta = ln.dgamma, ln.dbeta
    epsilon = 1e-5

    # -------------------------
    # Numerical dX
    # -------------------------

    numerical_dX = np.zeros_like(X)

    for i in range(B):
        for j in range(C):

            X_plus = X.copy()
            X_plus[i, j] += epsilon

            X_minus = X.copy()
            X_minus[i, j] -= epsilon

            loss_plus = (LayerNorm1d(C)(X_plus) * dout).sum()
            loss_minus = (LayerNorm1d(C)(X_minus) * dout).sum()

            numerical_dX[i, j] = (loss_plus - loss_minus) / (2 * epsilon)


    # -------------------------
    # Numerical dgamma
    # -------------------------

    numerical_dgamma = np.zeros_like(ln.gamma)

    for j in range(C):

        ln_plus = LayerNorm1d(C)
        ln_minus = LayerNorm1d(C)

        ln_plus.gamma[j] += epsilon
        ln_minus.gamma[j] -= epsilon

        loss_plus = (ln_plus(X) * dout).sum()
        loss_minus = (ln_minus(X) * dout).sum()

        numerical_dgamma[j] = (loss_plus - loss_minus) / (2 * epsilon)


    # -------------------------
    # Numerical dbeta
    # -------------------------

    numerical_dbeta = np.zeros_like(ln.beta)

    for j in range(C):

        ln_plus = LayerNorm1d(C)
        ln_minus = LayerNorm1d(C)

        ln_plus.beta[j] += epsilon
        ln_minus.beta[j] -= epsilon

        loss_plus = (ln_plus(X) * dout).sum()
        loss_minus = (ln_minus(X) * dout).sum()

        numerical_dbeta[j] = (loss_plus - loss_minus) / (2 * epsilon)


    # -------------------------
    # Results
    # -------------------------

    print("dX analytical:")
    print(dX)

    print("\ndX numerical:")
    print(numerical_dX)

    print("\ndX max difference:")
    print(np.max(np.abs(dX - numerical_dX)))

    print("\n")

    print("dgamma analytical:")
    print(dgamma)

    print("\ndgamma numerical:")
    print(numerical_dgamma)

    print("\ndgamma max difference:")
    print(np.max(np.abs(dgamma - numerical_dgamma)))

    print("\n")

    print("dbeta analytical:")
    print(dbeta)

    print("\ndbeta numerical:")
    print(numerical_dbeta)

    print("\ndbeta max difference:")
    print(np.max(np.abs(dbeta - numerical_dbeta)))