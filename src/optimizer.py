from .module import Module
import cupy as np
class Optimizer(Module):
    def __init__(self,params,lr,beta1=0.9,beta2=0.999,w_decay=0.0,eps=1e-8):
        self.params = params
        self.eps = eps
        self.lr = lr
        self.w_decay = w_decay
        self.ms = []
        self.vs = []
        for p in self.params:
            self.ms.append(np.zeros_like(p))
            self.vs.append(np.zeros_like(p))
                        
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
    def step(self,grads):
        self.t +=1
        for i,(p,g,m,v) in enumerate(zip(self.params,grads,self.ms,self.vs)):
            # m is m_{t-1}
            m_t = self.beta1 * m + (1-self.beta1)*g
            self.ms[i] = m_t
            v_t = self.beta2 * v + (1-self.beta2)*g**2
            self.vs[i] = v_t
            v_hat = v_t/(1-self.beta2**self.t)
            m_hat = m_t/(1-self.beta1**self.t)
            p -= self.lr*((m_hat / (np.sqrt(v_hat) + self.eps))+   self.w_decay * p)
        

if __name__ == "__main__":
 
    # One parameter
    p = np.array([1.0])

    # Adam with simple settings
    optimizer = Optimizer(
        params=[p],
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        w_decay=0.0,
        eps=1e-8
    )

    # Same gradient every step
    g = np.array([0.1])

    for step in range(3):

        optimizer.step([g])

        m = optimizer.ms[0]
        v = optimizer.vs[0]

        m_hat = m / (1 - optimizer.beta1 ** optimizer.t)
        v_hat = v / (1 - optimizer.beta2 ** optimizer.t)

        print(f"Step {step + 1}")
        print("m     =", m)
        print("v     =", v)
        print("m_hat =", m_hat)
        print("v_hat =", v_hat)
        print("p     =", p)
        print()



