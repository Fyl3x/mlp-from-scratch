from .backend import np
from .module import Module
class CrossEntropyLoss(Module):
    def __init__(self,logits,targets):
        self.logits = logits
        self.targets = targets
    def forward(self):
        max_logits = self.logits.max(axis=1,keepdims=True)
        self.max_indices = self.logits.argmax(axis=1,keepdims=True)
        stable_logits = self.logits - max_logits
        e_logits = np.exp(stable_logits)
        e_logits_sum = e_logits.sum(axis=1,keepdims=True)
        self.probs = e_logits / e_logits_sum
        logprobs = np.log(self.probs)  
        smoothing = 0.1
        n_classes = self.logits.shape[1]
        target_probs = np.full_like(self.probs, smoothing / n_classes)
        target_probs[np.arange(len(self.targets)), self.targets] += (1 - smoothing)
        
        loss = -(target_probs * logprobs).sum(axis=1).mean()        
        self.B = len(self.targets)
        self.logprobs = logprobs
        self.target_probs = target_probs

        self.e_logits_sum = e_logits_sum
        self.e_logits = e_logits
        self.stable_logits = stable_logits
        return loss
    def backward(self):
        dlogprobs = -self.target_probs / self.B
        # dlogprobs[np.arange(self.B),self.targets] = (-1/self.B)
        dprobs = dlogprobs * (1/self.probs)
        de_logits_sum = dprobs * self.e_logits * (-1/(self.e_logits_sum)**2)
        de_logits_sum = de_logits_sum.sum(axis=1,keepdims=True)
        de_logits = dprobs * 1/self.e_logits_sum + de_logits_sum
        dstable_logits = de_logits * self.e_logits
        dmax_logits = -dstable_logits.sum(axis=1,keepdims=True)
        ph = np.zeros_like(dstable_logits)
        ph[np.arange(self.B),self.max_indices.flatten()] = dmax_logits.flatten()
        dlogits = dstable_logits + ph
        return dlogits

