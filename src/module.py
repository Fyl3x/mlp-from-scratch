class Module:
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    def parameters(self):
        results = []
        return results
    def gradients(self):
        result = []
        return result