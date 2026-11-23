# 2.4 The maths (25 min)

## Why this matters
In lab 1, `loss.backward()` and `optimizer.step()` did the learning. Here's what they computed,
for the simplest model, a **linear classifier**:

1. **scores**: `z = X·W + b`, one weighted sum of the review's features per class (the logits).
   Lab 1's keyword baseline was this with hand-picked weights: +1 for "great", −1 for "rude".
2. **softmax** turns scores into probabilities: `p = exp(z) / sum(exp(z))`.
3. **cross-entropy**: loss = `−log(p of the true class)`, averaged. It's 0.01 for a confident right
   answer and 6.9 for a confident wrong one.
4. **gradient**: `dW = Xᵀ(P − Y) / n`, where `P − Y` is "predicted probability minus truth". Each
   word's weight moves in proportion to how wrong the predictions were on reviews containing it.
5. **gradient descent**: `W ← W − learning_rate · dW`, repeated.

That loop (forward → loss → gradient → step) is all of deep learning. Bigger models change step 1.

## Predict first (notes.md)
Loss of a model that answers 1/3 for every class? What happens with a learning rate that's far too big?

## Prompt
> Explain softmax, cross-entropy and their gradient to me with a tiny numeric example (3 classes,
> 2 reviews), then write `mathlab.py` in this folder with NumPy functions `softmax(Z)`,
> `cross_entropy(P, y)` (y = integer labels) and `gradients(X, P, Y)` returning `(dW, db)` (X may
> be a scipy sparse matrix, Y one-hot), plus `train_linear(X, y, lr, steps)` using plain gradient
> descent. Verify the gradient against PyTorch autograd on a small batch. Then train on our TF-IDF
> features (from `outputs/train.csv`) with learning rates [DECISION: try three, e.g. 1, 10, 25],
> plot the loss curves to `outputs/lr_curves.png`, and compare the best with lab 1's network on dev.

## Verify
`lab check 2.4` tests your functions, including against PyTorch's own gradients.
- Did the from-scratch linear model beat lab 1's network? What does that tell you?
- Ask Claude: *"Why do we subtract the max inside softmax?"* and *"Derive the gradient in five lines."*

## ML thinking
**Understand the simplest model that works before reaching for a bigger one.**

## Go deeper
logistic regression · softmax regression · maximum likelihood · backpropagation · automatic differentiation · learning rate schedules
