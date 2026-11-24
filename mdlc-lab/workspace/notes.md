# Lab journal

Circle your prediction **before** running, then write the result. One line each is plenty.

## Lab 1

### 1.1 Meet the data
- Predict: always-"positive" accuracy on dev?  A 33% · B 60% · C 85% →
- Result:

### 1.2 Baselines
- Predict: majority baseline macro-F1?  A 0.25 · B 0.50 · C 0.62 →
- Result:
- Metric from now on:  A accuracy · B macro-F1 →

### 1.3 Text to numbers
- Predict: non-zero slots per review?  A ~10 · B ~100 · C ~5,000 →
- Result:

### 1.4 Training
- Predict: dev loss over 6 epochs?  A keeps falling · B falls, then rises · C rises from the start →
- Result:

### 1.5 Experiments
- Predictions go in the `predicted` column of outputs/experiments.csv. Lesson:

### 1.6 Evaluate
- Predict: held-out vs dev macro-F1?  A clearly higher · B about the same · C clearly lower →
- Result:
- Neutral F1 on held-out:

### 1.7 Break it
- Predict: starter reviews that fool it?  A 0–2 · B 3–5 · C 6–8 →
- Result:

## Lab 2

### 2.1 Audit
- Predict: kinds of planted problem?  A 2–3 · B 4–6 · C 7+ →
- Dataset-level problems I found:

### 2.2 Clean and split
- Predict: rows surviving?  A ~20,500 · B ~19,900 · C ~18,000 →
- Result:

### 2.3 Labels
- Predict: 3★ golden reviews that read as neutral?  A ~80% · B ~50% · C ~30% →
- Result:

### 2.4 Maths
- Predict: loss of a model answering 1/3 for every class?  A 0.33 · B 1.10 · C 3.00 →
- Result:

### 2.5 Loss design
- Predict: neutral recall with inverse weights?  A falls · B about the same · C roughly doubles →
- Ship for support routing:  A none · B sqrt · C inverse →  because

### 2.6 Label noise
- Predict: planted flips among the top 600?  A ~100 · B ~300 · C ~550 →
- Suspects:  A drop · B down-weight · C relabel →

### 2.7 More data
- Predict: dev macro-F1 change, 18k → 100k?  A +0.00 · B +0.02 · C +0.05 or more →
- Result:

### 2.8 Synthetic data
- Predict: effect of 300 synthetic reviews?  A clearly helps · B no measurable change · C hurts →
- Result (from 2.9, C vs B):

### 2.9 Retrain and compare
- Predict: lab 2 vs lab 1 on held-out?  A no significant difference · B +0.01 to +0.03 · C +0.04 or more →
- Result:
