import pandas as pd 
import numpy as np
from src.embedding import Embedding
from src.vocab import Vocab
from src.mlp import MLPHead
from src.crossentropy import CrossEntropyLoss


embed_dim = 64
# importing the whole training data at once (this is text)
train_data = pd.read_csv("data/processed/train.csv")

# Preparing the Data Alongside its targets 
dictio = {}
for idx,product in enumerate(sorted(train_data["Product"].unique())):
    dictio[product] = idx
train_data['targets'] = train_data["Product"].map(dictio)    
def get_batch(batch_size):
    
    batch_sample = train_data.sample(n=batch_size)
    X_batch = batch_sample["Consumer complaint narrative"]
    y_batch = batch_sample["targets"]
    return X_batch,y_batch

# get 32 examples as a batch
sample,targets = get_batch(32) # this is still text
num_classes = len(train_data["targets"].unique()) # this gives 12 classes

# Initialization of the Vocab Object to use it
Vocabulary = Vocab(2,400,train_data)
Vocabed_text = np.array([Vocabulary.encode(text) for text in sample])
# these are now Numbers (32,400)

# Initialization of the Embedding Object
Embeddor = Embedding(len(Vocabulary.word2idx),embed_dim=embed_dim) # this is a (Vocab_size,64) Lookup Table
Embedded_text = Embeddor(Vocabed_text) # this is now (32,400,64) summed over dim = 1
# this is now (32,64)

# Initialization of the MLP Head
NeuralNet = MLPHead(embed_dim=embed_dim,output_dim=num_classes)
MLP_result = NeuralNet(Embedded_text) # this now goes through the follwing
#     Linear -> BatchNorm -> Relu -> Linear -> BatchNorm -> Relu -> Linear -> (32,12)
# 32,64 -> 32,256               32,256 -> 32,64                32,64 -> 32,12 (logits)


loss = CrossEntropyLoss()
losses = loss(MLP_result,targets)
print(losses)
