import re
import math
import random
from collections import Counter

from naive_bayes import EvaluationMetrics, load_imdb_dataset, train_test_split, run_evaluation, IMDBNaiveBayesClassifier
 

_tokenizer = IMDBNaiveBayesClassifier()
 
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(_tokenizer.tokenize(text))
    return {word: idx for idx, (word, _) in enumerate(counter.most_common(3000))}
 
def vectorize(text, vocab):
    vec = [0] * len(vocab)
    for token in _tokenizer.tokenize(text):
        if token in vocab:
            vec[vocab[token]] = 1
    return vec
 
 
w = []
b = 0
 
def predict(x):
    score = sum(w[i] * x[i] for i in range(len(x))) + b
    if score > 0:
        return "positive"
    else:
        return "negative"
 
def update(x, y):
    global w, b
    y_num = 1 if y == "positive" else -1
    pred_num = 1 if predict(x) == "positive" else -1
    if pred_num != y_num:
        for i in range(len(w)):
            w[i] += y_num * x[i]
        b += y_num
 
 
# loads the dataset
print("Loading dataset...")
texts, labels = load_imdb_dataset("IMDB Dataset.csv")
 
# uses a subset for training
random.seed(42)
indices = list(range(len(texts)))
random.shuffle(indices)
indices = indices[:5000]
texts  = [texts[i]  for i in indices]
labels = [labels[i] for i in indices]
 
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, seed=42)
 
print("Building vocabulary...")
vocab = build_vocab(x_train)
 
print("Vectorising...")
train_data = [(vectorize(t, vocab), lbl) for t, lbl in zip(x_train, y_train)]
test_data  = [(vectorize(t, vocab), lbl) for t, lbl in zip(x_test,  y_test)]
 
# initialize weights
w = [0.0] * len(vocab)
b = 0
 
# training loop
epochs = 10
print("Training...\n")
for _ in range(epochs):
    for x, y in train_data:
        update(x, y)
 
# evaluate using previous homework evaluation code
y_pred = [predict(vectorize(t, vocab)) for t in x_test]
print("=== Evaluation ===")
run_evaluation(y_test, y_pred)
 
# test predictions
tests = ["This movie was wonderful", "Terrible and boring film", "Amazing cast and story"]
print("\n=== Predictions ===")
for t in tests:
    print(t, "->", predict(vectorize(t, vocab)))
 
# inspect weights
word_weights = sorted([(word, w[idx]) for word, idx in vocab.items()], key=lambda x: x[1], reverse=True)
 
print("\nWhich words have positive weights?")
for word, weight in word_weights[:10]:
    print(f"  {word}: {weight:.1f}")
 
print("\nWhich words have negative weights?")
for word, weight in word_weights[-10:][::-1]:
    print(f"  {word}: {weight:.1f}")

