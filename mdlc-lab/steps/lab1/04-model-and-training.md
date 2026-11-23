# 1.4 The model and the training loop (25 min)

## Why this matters
A **neural network** is a function with many adjustable numbers called **weights**. Ours:
`TF-IDF vector → Linear → ReLU → Linear → 3 scores (logits) → softmax → 3 probabilities`.

**Training** repeats a loop:
1. take a **batch** of reviews (e.g. 64);
2. **forward pass**: compute predictions;
3. compute the **loss**: one number for how wrong they were. We use **cross-entropy**: roughly,
   how little probability the model gave the right answer;
4. **backward pass**: compute the **gradient**, the direction to nudge each weight to reduce the loss;
5. the **optimizer** moves each weight a small step (the **learning rate**).

One pass over the training data is an **epoch**. After each epoch, measure the loss on **dev** too.
When dev loss starts rising while train loss keeps falling, the model is **overfitting**: memorising
training reviews instead of learning patterns. **Early stopping** keeps the weights from the best
dev epoch.

## Predict first (notes.md)
Sketch the train-loss and dev-loss curves over 6 epochs. Where do they diverge?

## Prompt
> Write `src/train.py` using PyTorch and the features from `src/features.py`: a network
> Linear(vocab → [DECISION: hidden size]) → ReLU → Linear(→ 3), cross-entropy loss, Adam optimizer
> with learning rate [DECISION], batches of 64, 6 epochs. After every epoch log epoch, train_loss,
> dev_loss, train_acc and dev_acc to `outputs/training_log.csv`, and save a plot of the curves to
> `outputs/training_curves.png`. Keep the weights from the epoch with the lowest dev loss. Write
> that model's dev predictions to `outputs/dev_network.csv` (`id,label`) and save the model and
> vectorizer to `outputs/` for later steps. Walk me through the five lines of the training step.

## Verify
`lab check 1.4` · `lab score outputs/dev_network.csv`
- Open `outputs/training_curves.png`. Where did overfitting start? Did your sketch match?
- Ask Claude: *"Which data was the model trained on, and which was used only for measuring? Point to
  the code."*
- Ask: *"How many weights does this network have?"*

## ML thinking
**Training loss tells you about memory; dev loss tells you about learning.** Always watch both.

## Go deeper
cross-entropy loss · gradient descent · backpropagation · Adam optimizer · overfitting · early stopping · learning rate
