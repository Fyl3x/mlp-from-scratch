import argparse
import copy
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd 
from src.vocab import Vocab
import json,os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
torch.manual_seed(42)

class MLPHead(nn.Module):
    def __init__(self,vocab_size,n_embd,input_dim,output_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,n_embd)
        nn.init.normal_(self.embedding_layer.weight, std=0.01)  # match Phase 1
        self.linear1 = nn.Linear(input_dim,4*input_dim,bias=False)
        self.bn1 = nn.BatchNorm1d(4*input_dim,momentum=0.3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4*input_dim,input_dim,bias=False)
        self.bn2 = nn.BatchNorm1d(input_dim,momentum=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
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
test_data = pd.read_csv("data/processed/test.csv")
test_data["targets"] = test_data["Product"].map(dictio)
assert test_data["targets"].notna().all(), "test has a label unseen in train"

epoch_size = len(train_data["Consumer complaint narrative"])
num_classes = len(train_data["targets"].unique())
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
num_epochs = parser.parse_args().epochs


model = MLPHead(vocab_size=vocab_size,n_embd=n_embd,input_dim=n_embd,output_dim=num_classes).to(device=device)
@torch.no_grad()
def predict_logits(X, batch_size=1024):
    return torch.cat([model(X[s:s + batch_size].to(device)) for s in range(0, len(X), batch_size)])
optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3,weight_decay=1e-3,betas=(0.9,0.999),eps=1e-8)

# Training set
Whole_vocab = torch.LongTensor([vocabulary.encode(text) for text in train_data["Consumer complaint narrative"]])
whole_targets = torch.LongTensor(train_data["targets"])

# Validation set
val_vocab = torch.LongTensor([vocabulary.encode(text) for text in val_data["Consumer complaint narrative"]])
val_targets = torch.LongTensor(val_data["targets"])
# Testing set
test_vocab = torch.LongTensor([vocabulary.encode(text) for text in test_data["Consumer complaint narrative"]])
test_targets = torch.LongTensor(test_data["targets"])

# Tracking measures
train_losses, val_losses, val_accuracies = [], [], []

# Checkpointing 
best_val = float('inf')
best_epoch = None
best_state = None
# Training Loop
for e in range(num_epochs): 
    epoch_loss_sum = 0
    
    perm = torch.randperm(epoch_size)
    for i in range(epoch_size//batch_size):
        start = i * batch_size
        end = start + batch_size
        idx = perm[start:end]

        xb = Whole_vocab[idx].to(device)
        targets = whole_targets[idx].to(device)
        forward = model(xb)
        loss = F.cross_entropy(forward,target=targets,label_smoothing=0.1)
        epoch_loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"epoch {e} iter {i}: loss {loss.item():.4f}")

    train_losses.append(epoch_loss_sum / (epoch_size//batch_size))
    model.eval()
    with torch.no_grad():
        targets = val_targets.to(device)
        forward = predict_logits(val_vocab)
        loss = F.cross_entropy(forward,target=targets,label_smoothing=0.1)
        current_val_loss = loss.item()
        maxes = torch.argmax(forward,axis=1)
        accuracy = (maxes == targets).float().mean() * 100
        val_losses.append(loss.item())
        val_accuracies.append(accuracy.item())
        print(f"epoch {e}: val_loss = {loss.item():.4f}, val_accuracy = {accuracy.item():.2f}%")
        if current_val_loss < best_val:
            best_val = current_val_loss
            best_epoch = e + 1
            best_state = copy.deepcopy(model.state_dict())
        
    model.train()

model.load_state_dict(best_state)
print(f"best val loss (restored): {best_val}")
model.eval()  # the loop ended with model.train(): same trap as Phase 1
test_logits = predict_logits(test_vocab)
y = test_targets.to(device)
test_loss = F.cross_entropy(test_logits, y, label_smoothing=0.1).item()
test_preds = test_logits.argmax(dim=1)
test_acc = (test_preds == y).float().mean().item() * 100
print(f"test_acc = {test_acc:.2f}%, test_loss (label-smoothed CE) = {test_loss:.4f}")

y_true, y_pred = test_targets.numpy(), test_preds.cpu().numpy()
macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
print(f"test macro F1 = {macro_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=list(dictio.keys()), digits=3))
results = {
    "phase": "torch",
    "backend": f"torch-{device}",
    "config": {
        "epochs": num_epochs, "batch_size": batch_size, "lr": 1e-3, "weight_decay": 1e-3,
        "embed_dim": n_embd, "min_count": min_count, "max_len": max_length, "seed": 42,
        "dropout": 0.5, "bn_momentum": 0.3, "label_smoothing": 0.1,
        "note": "bn_momentum uses torch's convention; equals 0.7 in Phase 1's convention",
    },
    "labels": list(dictio.keys()),
    "history": {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accuracies},
    "best_epoch": best_epoch,
    "test": {
        "acc": test_acc, "loss_label_smoothed": test_loss, "macro_f1": macro_f1,
        "per_class": classification_report(y_true, y_pred, target_names=list(dictio.keys()), output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    },
}
os.makedirs("results/torch", exist_ok=True)
with open("results/torch/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results/torch/metrics.json")