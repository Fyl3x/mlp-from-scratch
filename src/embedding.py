import numpy as np
import pandas as pd 
from .vocab import Vocab 
from .module import Module
class Embedding(Module):
    def __init__(self,vocab_size,embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = np.random.randn(self.vocab_size,self.embed_dim) * 0.01

    def forward(self,batch):
        result = self.weight[batch] # this gives B,T, embed_dim
        result = result.sum(axis = -2) # this is now B,C     
        return result
    def backward(self,grad_output):
        pass

if __name__ == "__main__":
    test_data = pd.read_csv("./data/processed/train.csv")

    vocabulary = Vocab(2,400,test_data["Consumer complaint narrative"])
    emb = Embedding(vocab_size=len(vocabulary.word2idx), embed_dim=64)
    # batch = np.array([vocabulary.encode(narrative)])  # shape should be (1, 400)
    # out = emb.forward(batch)
    # print(batch.shape)   # expect (1, 400)
    # print(out.shape)     # expect (1, 64)
    

    sample = test_data["Consumer complaint narrative"].iloc[:5]
    batch = np.array([vocabulary.encode(text) for text in sample])
    print(batch.shape)   # expect (5, 400)

    out = emb.forward(batch)
    print(out.shape)     # expect (5, 64)