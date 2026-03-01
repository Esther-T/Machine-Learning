import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter


# -----------------start of helper functions---------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def f(x, theta):
    return sigmoid(x @ theta)

def log_likelihood(theta, x, y):
    y_ = f(x, theta)
    eps = 1e-6
    N = len(y)
    return (1/N) * np.sum(y* np.log(y_ + eps) + (1 - y) * np.log(1 - y_ + eps))

def grad(theta, x, y):
    N = len(y)
    return (1/N) * x.T @ (f(x, theta) - y)
    
#building vocab aka bag of words
def build_vocab(prompts):
    vocab = set()
    for prompt in prompts:
        words = prompt.lower().split()
        vocab.update(words)
    return sorted(vocab)  

#vectorize each word 
def vectorize(prompts, vocab):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    X = np.zeros((len(prompts), len(vocab)))
    for i, prompt in enumerate(prompts):
        counts = Counter(prompt.lower().split())
        for word, count in counts.items():
            if word in vocab_index:
                X[i, vocab_index[word]] = count
    return X    
    
    
# -----------------end of helper functions---------------------------------

# Load datasets
with open("dataset.json", "r") as file:
    data = json.load(file)

prompts = [entry["prompt"] for entry in data]
labels = np.array([entry["label"] for entry in data])


# Data Prep
vocab = build_vocab(prompts)
X = vectorize(prompts, vocab)
labels = np.array(labels)

print(vocab)
print(X)

#The features would depend of the vocabulary words (eg: each unique word is a feature)

# I will implement the training loop manually and plot the results
theta = np.zeros(X.shape[1])
learning_rate = 0.1 #recommended for small datasets
num_iterations = 1000

for i in range(num_iterations):
    theta -= learning_rate * grad(theta, X, labels)
    if i % 100 == 0:
        print(f"Iteration {i}, Log-Likelihood: {log_likelihood(theta, X, np.array(labels))}")
    # I want to check the log likelhood after 100 iterations

predictions = (f(X, theta) >= 0.5).astype(int)
print("Predictions: ", predictions)
print("Actuals: ", labels)
# Yes! Looks great!

# Now I'm going to evaluate using a different set of test data

# Load test datasets
with open("testdataset.json", "r") as file:
    data = json.load(file)

test_prompts = [entry["prompt"] for entry in data]
test_labels = np.array([entry["label"] for entry in data])

X_test = vectorize(test_prompts, vocab)
test_predictions = (f(X_test, theta) >= 0.5).astype(int)
print("(Test Data) Predictions:", test_predictions)
print("(Test Data) Actuals:    ", test_labels)
