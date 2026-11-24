# 2.4 The maths (20 min, or an instructor demo if time is short)
**Goal:** see what `loss.backward()` and `optimizer.step()` actually compute.

## Why this matters
A **linear classifier**: scores `z = X·W + b` → **softmax** → probabilities → **cross-entropy**
`−log p(true class)` → **gradient** `dW = Xᵀ(P − Y)/n` → step `W ← W − lr·dW`. That loop is all of deep learning.

## Predict
Loss of a model that answers 1/3 for every class?  **A** 0.33  **B** 1.10  **C** 3.00

## Prompt
> Explain softmax, cross-entropy and the gradient with a tiny example (3 classes, 2 reviews). Then
> write `mathlab.py` in this folder with NumPy functions `softmax(Z)`, `cross_entropy(P, y)`
> (`y` integer labels), `gradients(X, P, Y)` → `(dW, db)` (`X` may be scipy sparse, `Y` one-hot), and
> `train_linear(X, y, lr, steps)` (plain gradient descent). Check the gradient against PyTorch autograd.
> Train on TF-IDF features from `outputs/train.csv` for 1,500 steps.
> [DECISION: learning rates] **A** 1, 10, 25 (default) · **B** 0.1, 1, 10 · **C** 10, 100, 1000.
> Plot the loss curves to `outputs/lr_curves.png` and report the best one's dev macro-F1.

## Verify
`lab check 2.4` · Does any curve zigzag? Ask Claude why (hint: look at the bias `b`).

## Takeaway
Understand the simplest model that works before reaching for a bigger one.

## Go deeper
logistic regression · backpropagation · autodiff · learning rate schedules ·
*Ask Claude: "Why subtract the max inside softmax?"* · *"Derive the gradient in five lines."*
