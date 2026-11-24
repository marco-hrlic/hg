# 1.4 The model and the training loop (25 min)
**Goal:** a trained network, its loss curves, and saved weights.

## Why this matters
Network: `TF-IDF → Linear → ReLU → Linear → 3 scores` (ReLU, rectified linear unit: keeps positives,
zeroes negatives). **Training loop**: batch → forward →
**loss** (cross-entropy) → backward (**gradient**) → optimizer step. One pass over train = an **epoch**.
Dev loss rising while train loss falls = **overfitting**; **early stopping** keeps the best epoch.

## Predict
Over 6 epochs, dev loss will…  **A** keep falling  **B** fall, then rise after epoch 1–3  **C** rise from the start

## Prompt
> Write `src/train.py` (PyTorch, features from `src/features.py`): Linear(vocab → hidden) → ReLU →
> Linear(→ 3), cross-entropy, Adam (an optimizer that adapts each weight's step size), batches of 64,
> 6 epochs. Before training, print an ASCII diagram of the network: each layer, its size, and its
> number of weights.
> [DECISION: hidden size] **A** 16 · **B** 32 (default) · **C** 256.
> [DECISION: learning rate] **A** 0.0001 · **B** 0.001 (default) · **C** 0.01.
> Log epoch, train_loss, dev_loss, train_acc, dev_acc to `outputs/training_log.csv`; plot them to
> `outputs/training_curves.png`. Keep the epoch with the lowest dev loss; write its dev predictions to
> `outputs/dev_network.csv` (`id,label`). Save `outputs/model.pt` and `outputs/vectorizer.pkl` (pickle).

## Verify
`lab check 1.4` · `lab score outputs/dev_network.csv` · Open the plot: where did dev loss turn?
In the diagram: which layer holds almost all the weights?

## Takeaway
Train loss tells you about memory, dev loss about learning. Watch both.

## Go deeper
cross-entropy · gradient descent · backpropagation · Adam · early stopping ·
*Ask Claude: "Walk me through the five lines of the training step."* ·
*"Why are almost all the weights in the first layer?"*
