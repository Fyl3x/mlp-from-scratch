import copy
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd 
from src.vocab import Vocab

device = 'cuda'
torch.manual_seed(42)

class MLPHead(nn.Module):
    def __init__(self,vocab_size,n_embd,input_dim,output_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,n_embd)
        self.linear1 = nn.Linear(input_dim,4*input_dim,bias=False)
        self.bn1 = nn.BatchNorm1d(4*input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(4*input_dim,input_dim,bias=False)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(input_dim,output_dim)
        self.sequence = nn.Sequential(self.linear1,
                           self.bn1,
                           self.relu,
                           self.dropout,
                           self.linear2,
                           self.bn2,
                           self.relu,
                           self.dropout2,
                           self.linear3)
        
    def forward(self,x):
        embedded_x = self.embedding_layer(x).sum(axis=1)
        result = self.sequence(embedded_x)
        return result
batch_size = 32
max_length = 400
n_embd = 64
min_count = 10


train_data = pd.read_csv("data/processed/train.csv")
val_data = pd.read_csv("data/processed/val.csv")
vocabulary = Vocab(min_count=min_count,max_len=max_length,texts=train_data["Consumer complaint narrative"])
vocab_size = len(vocabulary.word2idx)



dictio = {}
for idx, product in enumerate(sorted(train_data["Product"].unique())):
    dictio[product] = idx

train_data["targets"] = train_data["Product"].map(dictio)
val_data["targets"] = val_data["Product"].map(dictio)

epoch_size = len(train_data["Consumer complaint narrative"])
num_classes = len(train_data["targets"].unique())
num_epochs = 30


model = MLPHead(vocab_size=vocab_size,n_embd=n_embd,input_dim=n_embd,output_dim=num_classes)
optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3,weight_decay=1e-3,betas=(0.9,0.999),eps=1e-8)

# Training set
Whole_vocab = torch.LongTensor([vocabulary.encode(text) for text in train_data["Consumer complaint narrative"]])
whole_targets = torch.LongTensor(train_data["targets"])

# Validation set
val_vocab = torch.LongTensor([vocabulary.encode(text) for text in val_data["Consumer complaint narrative"]])
val_targets = torch.LongTensor(val_data["targets"])

# Tracking measures
train_losses, val_losses, val_accuracies = [], [], []

# Checkpointing 
best_val = float('inf')
best_state = None
# Training Loop
for _ in range(num_epochs): 
    epoch_loss_sum = 0
    
    perm = torch.randperm(epoch_size)
    for i in range(epoch_size//batch_size):
        start = i * batch_size
        end = start + batch_size
        idx = perm[start:end]

        input = Whole_vocab[idx]
        targets = whole_targets[idx]
        forward = model(input)
        loss = F.cross_entropy(forward,target=targets)
        epoch_loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"Loss at epoch {_}, at iteration {i} is : {loss.item()}")
    
    train_losses.append(epoch_loss_sum / (epoch_size//batch_size))
    model.eval()
    with torch.no_grad():
        input = val_vocab
        targets = val_targets
        forward = model(input)
        loss = F.cross_entropy(forward,target=targets)
        current_val_loss = loss.item()
        maxes = torch.argmax(forward,axis=1)
        accuracy = (maxes == targets).float().mean() * 100
        val_losses.append(loss.item())
        val_accuracies.append(accuracy.item())
        print(f"Epoch {_} Val Loss is : {loss.item()} with accurary: {accuracy} %")
    if current_val_loss < best_val:
            best_val = current_val_loss
            best_state = copy.deepcopy(model.state_dict())
        
    model.train()

model.load_state_dict(best_state)
print(f"best val loss (restored): {best_val}")

import matplotlib.pyplot as plt 
epochs_range = range(1, num_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_range, train_losses, label="Train Loss")
ax1.plot(epochs_range, val_losses, label="Validation Loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs_range, val_accuracies, label="Validation Accuracy")
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")

plt.tight_layout()
plt.show()











