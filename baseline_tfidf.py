import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from sklearn.metrics import f1_score, classification_report

TEXT, LABEL = "Consumer complaint narrative", "Product"
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

vec = TfidfVectorizer(lowercase=True, min_df=10, sublinear_tf=True)
X_train = vec.fit_transform(train[TEXT])  # fit on train only
X_val = vec.transform(val[TEXT])
X_test = vec.transform(test[TEXT])
print("shapes:", X_train.shape, X_val.shape, X_test.shape)
print("vocab size:", len(vec.vocabulary_))

best = None
for C in [0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_train, train[LABEL])
    acc = accuracy_score(val[LABEL], clf.predict(X_val)) * 100
    print(f"C={C}: val_acc = {acc:.2f}%")
    if best is None or acc > best[0]:
        best = (acc, C, clf)
print(f"best C = {best[1]} (val_acc = {best[0]:.2f}%)")


val_acc, C, clf = best
test_pred = clf.predict(X_test)
test_acc = accuracy_score(test[LABEL], test_pred) * 100
macro_f1 = f1_score(test[LABEL], test_pred, average="macro")
print(f"test_acc = {test_acc:.2f}%, test macro F1 = {macro_f1:.4f}")
print(classification_report(test[LABEL], test_pred, digits=3))

with open("results/baseline_tfidf.json", "w") as f:
    json.dump({
        "model": "TF-IDF (unigrams) + logistic regression",
        "config": {"min_df": 10, "sublinear_tf": True, "C_grid": [0.1, 1.0, 10.0], "best_C": C},
        "vocab_size": len(vec.vocabulary_),
        "val_acc": val_acc,
        "test": {"acc": test_acc, "macro_f1": float(macro_f1),
                "per_class": classification_report(test[LABEL], test_pred, output_dict=True)},
    }, f, indent=2)
print("saved results/baseline_tfidf.json")