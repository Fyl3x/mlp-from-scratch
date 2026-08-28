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
        self.batch = batch
        result = self.weight[batch] # this gives B,T, embed_dim
        result_summed = result.sum(axis = -2) # this is now B,C     
        return result_summed
    def backward(self,local_grad):
        B,T = self.batch.shape
        dresult_summed = local_grad # B,C
        dresult = np.broadcast_to(dresult_summed[:,np.newaxis,:],(B,T,self.embed_dim)) # this needs to be B,T,C
        dW = np.zeros_like(self.weight)
        # for i in range(B):
        #     for j in range(T):
        #         dW[self.batch[i,j],:] += dresult[i,j,:]
        np.add.at(
            dW,
            self.batch.reshape(-1),
            dresult.reshape(-1,self.embed_dim)
        )
        
        self.dW = dW
        return dW
    def parameters(self):
        params = super().parameters()
        params.append(self.weight)
        return params
    def gradients(self):
        result = super().gradients()
        result.append(self.dW)
        return result
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

    import numpy as np

    np.random.seed(42)

    vocab_size = 5
    embed_dim = 3

    embedding = Embedding(vocab_size, embed_dim)

    batch = np.array([
        [0, 2, 1, 2],
        [3, 2, 0, 4]
    ])

    dout = np.random.randn(2, 3)

    # Analytical
    embedding(batch)
    dW = embedding.backward(dout)

    # Numerical
    epsilon = 1e-5
    numerical_dweight = np.zeros_like(embedding.weight)

    for i in range(vocab_size):
        for j in range(embed_dim):

            original_value = embedding.weight[i, j]

            embedding.weight[i, j] = original_value + epsilon
            loss_plus = (embedding(batch) * dout).sum()

            embedding.weight[i, j] = original_value - epsilon
            loss_minus = (embedding(batch) * dout).sum()

            embedding.weight[i, j] = original_value

            numerical_dweight[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    print("Output shape:", embedding(batch).shape)

    print("\nAnalytical dWeight:")
    print(dW)

    print("\nNumerical dWeight:")
    print(numerical_dweight)

    print("\nDifference:")
    print(np.abs(dW - numerical_dweight))

    print("\nMax difference:")
    print(np.max(np.abs(dW - numerical_dweight)))