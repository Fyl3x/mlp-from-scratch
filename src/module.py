class Module:
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)