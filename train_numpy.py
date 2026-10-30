import argparse
import pandas as pd
# from src.backend import np
from src.embedding import Embedding
from src.vocab import Vocab
from src.mlp import MLPHead
from src.crossentropy import CrossEntropyLoss
from src.optimizer import Optimizer
import json, os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from src.backend import np, to_numpy, BACKEND
np.random.seed(42)
embed_dim = 64

# Load data
train_data = pd.read_csv("data/processed/train.csv")
val_data = pd.read_csv("data/processed/val.csv")

# Map Product labels -> integer targets (same mapping applied to both splits)
dictio = {}
for idx, product in enumerate(sorted(train_data["Product"].unique())):
    dictio[product] = idx

train_data['targets'] = train_data["Product"].map(dictio)
val_data['targets'] = val_data["Product"].map(dictio)
test_data = pd.read_csv("data/processed/test.csv")
test_data["targets"] = test_data["Product"].map(dictio)
assert test_data["targets"].notna().all(), "test has a label unseen in train"

num_classes = len(train_data["targets"].unique())  # 12 classes

# Vocab: word-level, min_count=10, sequences padded/truncated to max_len=400
Vocabulary = Vocab(10, 400, train_data["Consumer complaint narrative"])

# Model
Embeddor = Embedding(len(Vocabulary.word2idx), embed_dim=embed_dim)
NeuralNet = MLPHead(embed_dim=embed_dim, output_dim=num_classes)

params = NeuralNet.parameters() + Embeddor.parameters()
optim = Optimizer(params=params, lr=1e-3, w_decay=1e-3)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()
epochs = args.epochs
batch_size = 32
epoch_size = len(train_data["Consumer complaint narrative"])

best_val = float('inf')
best_params = None
best_epoch = None
train_losses, val_losses, val_accuracies = [], [], []

# Pre-encode text -> token IDs once, outside the training loop, so each
# epoch just shuffles/slices integer arrays instead of re-tokenizing text
Whole_Vocab = np.array([Vocabulary.encode(text) for text in train_data["Consumer complaint narrative"]])
Whole_targets = np.asarray(train_data["targets"].to_numpy())

Vocabed_val = np.array([Vocabulary.encode(text) for text in val_data["Consumer complaint narrative"]])
target = np.asarray(val_data["targets"].to_numpy())
Vocabed_test = np.array([Vocabulary.encode(text) for text in test_data["Consumer complaint narrative"]])
test_target = np.asarray(test_data["targets"].to_numpy())
print("train/val/test:", Whole_Vocab.shape, Vocabed_val.shape, Vocabed_test.shape)

def set_train_mode(flag):
    for layer in (NeuralNet.bn1, NeuralNet.bn2, NeuralNet.dropout1, NeuralNet.dropout2):
        layer.training = flag
def predict_logits(X, batch_size=1024):
    chunks = []
    for s in range(0, len(X), batch_size):
        chunks.append(NeuralNet(Embeddor(X[s:s + batch_size])))
    return np.concatenate(chunks)
for e in range(epochs):
    epoch_loss_sum = 0
    perm = np.random.permutation(epoch_size)  # fresh shuffle order each epoch

    for i in range(epoch_size // batch_size):
        start = i * batch_size
        end = start + batch_size
        batched_indices = perm[start:end]

        Vocabed_text = Whole_Vocab[batched_indices]
        targets = Whole_targets[batched_indices]

        Embedded_text = Embeddor(Vocabed_text)  # (B, 400) -> (B, 64), sum-pooled

        forward = NeuralNet(Embedded_text)  # (B, 64) -> (B, 12) logits
        loss = CrossEntropyLoss(forward, targets=targets)
        loss_val = loss()
        epoch_loss_sum += loss_val

        dlogits = loss.backward()
        dEmbedded = NeuralNet.backward(dlogits=dlogits)
        dEmbedding = Embeddor.backward(dEmbedded)
        grads = NeuralNet.gradients() + [dEmbedding]
        optim.step(grads=grads)

        if i % 500 == 0:
            print(f"epoch {e} iter {i}: loss {float(loss_val):.4f}")

    train_losses.append(float(epoch_loss_sum / (epoch_size // batch_size)))

    # Validation: BatchNorm switches to running stats, not batch stats
    set_train_mode(False)

    
    forward_val = predict_logits(Vocabed_val)
    predictions = np.argmax(forward_val, axis=1)
    accuracy = (predictions == target).mean() * 100

    val_loss = CrossEntropyLoss(forward_val, targets=target)
    current_val_loss = val_loss()

    val_losses.append(float(current_val_loss))
    val_accuracies.append(float(accuracy))
    print(f"epoch {e}: val_loss = {float(current_val_loss):.4f}, val_accuracy = {accuracy:.2f}%")

    if current_val_loss < best_val:
        best_val = current_val_loss
        best_params = [p.copy() for p in NeuralNet.parameters() + Embeddor.parameters()]
        best_epoch = e + 1
    set_train_mode(True)

# Restore whichever epoch had the best val loss, not just the last one
all_params = NeuralNet.parameters() + Embeddor.parameters()
for p, best_p in zip(all_params, best_params):
    p[...] = best_p

print(f"best val loss (restored): {best_val}")
set_train_mode(False)
test_logits = predict_logits(Vocabed_test)
test_preds = np.argmax(test_logits, axis=1)
test_acc = float((test_preds == test_target).mean() * 100)
test_loss = float(CrossEntropyLoss(test_logits, targets=test_target)())
print(f"test_acc = {test_acc:.2f}%, test_loss (label-smoothed CE) = {test_loss:.4f}")
y_true, y_pred = to_numpy(test_target), to_numpy(test_preds)
macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
print(f"test macro F1 = {macro_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=list(dictio.keys()), digits=3))

results = {
    "phase": "numpy",
    "backend": BACKEND,
    "config": {
        "epochs": epochs, "batch_size": batch_size, "lr": 1e-3, "weight_decay": 1e-3,
        "embed_dim": embed_dim, "min_count": 10, "max_len": 400, "seed": 42,
        "dropout": 0.5, "bn_momentum": 0.7, "label_smoothing": 0.1,
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
os.makedirs("results/numpy", exist_ok=True)
with open("results/numpy/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results/numpy/metrics.json")
