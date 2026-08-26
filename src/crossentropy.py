import numpy as np 
from .module import Module
class CrossEntropyLoss(Module):
    def __init__(self):
        pass
    def forward(self,logits,targets):
        max_logits = logits.max(axis=1,keepdims=True)
        stable_logits = logits - max_logits
        e_logits = np.exp(stable_logits)
        normalized_logits = e_logits / e_logits.sum(axis=1,keepdims=True)
        loss = -np.log(normalized_logits[np.arange(len(targets)),targets]).mean()
        return loss
    def backward(self):
        pass