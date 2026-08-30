import cupy as np
from .module import Module


class ReLU(Module):
    def __init__(self):
        pass
    def forward(self,x):
        self.out = np.maximum(x,0)
        self.mask = (x > 0)
        return self.out
    def backward(self,local_grad):
        self.local_grad = local_grad
        # drelu = np.zeros_like(self.out)
        # drelu[self.mask] = local_grad
        drelu = local_grad * self.mask
        return drelu
    def parameters(self):
        return super().parameters()
    def gradients(self):
        return super().gradients()


if __name__ == "__main__":
    test_arr = np.array([[1.0,-5.0,17.0]])
    relu = ReLU()
    test_result = relu(test_arr)
    print(test_result)

