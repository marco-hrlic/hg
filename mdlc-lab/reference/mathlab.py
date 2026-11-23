"""Reference for lab 2 step 4: the maths of a linear classifier, in NumPy."""

import numpy as np


def softmax(Z):
    """Rows of scores -> rows of probabilities. Subtracting the row max avoids overflow."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, y):
    """Mean of -log(probability given to the true class)."""
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def gradients(X, P, Y):
    """dL/dW = X^T (P - Y) / n,  dL/db = mean of (P - Y)."""
    D = (P - Y) / X.shape[0]
    return np.asarray(X.T @ D), D.sum(axis=0)


def train_linear(X, y, lr=10.0, steps=1500):
    Y = np.eye(3)[y]
    W, b = np.zeros((X.shape[1], 3)), np.zeros(3)
    history = []
    for step in range(steps):
        P = softmax(np.asarray(X @ W) + b)
        history.append(cross_entropy(P, y))
        dW, db = gradients(X, P, Y)
        W -= lr * dW
        b -= lr * db
    return W, b, history
