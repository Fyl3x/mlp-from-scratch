import argparse
import pandas as pd
from src.backend import np
from src.embedding import Embedding
from src.vocab import Vocab
from src.mlp import MLPHead
from src.crossentropy import CrossEntropyLoss
from src.optimizer import Optimizer
import matplotlib.pyplot as plt

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
train_losses, val_losses, val_accuracies = [], [], []

# Pre-encode text -> token IDs once, outside the training loop, so each
# epoch just shuffles/slices integer arrays instead of re-tokenizing text
Whole_Vocab = np.array([Vocabulary.encode(text) for text in train_data["Consumer complaint narrative"]])
Whole_targets = np.asarray(train_data["targets"].to_numpy())

Vocabed_val = np.array([Vocabulary.encode(text) for text in val_data["Consumer complaint narrative"]])
target = np.asarray(val_data["targets"].to_numpy())

def set_train_mode(flag):
    for layer in (NeuralNet.bn1, NeuralNet.bn2, NeuralNet.dropout1, NeuralNet.dropout2):
        layer.training = flag

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

        if i % 20 == 0:
            print(loss_val)

    train_losses.append(float(epoch_loss_sum / (epoch_size // batch_size)))

    # Validation: BatchNorm switches to running stats, not batch stats
    set_train_mode(False)

    Embedded_val = Embeddor(Vocabed_val)
    forward_val = NeuralNet(Embedded_val)
    predictions = np.argmax(forward_val, axis=1)
    accuracy = (predictions == target).mean() * 100

    val_loss = CrossEntropyLoss(forward_val, targets=target)
    current_val_loss = val_loss()

    val_losses.append(float(current_val_loss))
    val_accuracies.append(float(accuracy))
    print(f"epoch {e}: val_loss = {current_val_loss}, val_accuracy = {accuracy:.2f}%")

    if current_val_loss < best_val:
        best_val = current_val_loss
        best_params = [p.copy() for p in NeuralNet.parameters() + Embeddor.parameters()]

    set_train_mode(True)

# Restore whichever epoch had the best val loss, not just the last one
all_params = NeuralNet.parameters() + Embeddor.parameters()
for p, best_p in zip(all_params, best_params):
    p[...] = best_p

print(f"best val loss (restored): {best_val}")

epochs_range = range(1, epochs + 1)
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