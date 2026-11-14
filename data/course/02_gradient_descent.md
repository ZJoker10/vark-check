# Lecture 2 — Gradient Descent

Course: Introduction to Machine Learning (demo material for VARK Check).

## 2.1 The optimisation problem

Training a linear model means finding parameters that minimise the loss. For
the linear model in Lecture 1 the loss is a convex quadratic bowl in `w` and
`b`, so a closed-form solution exists. Gradient descent is introduced anyway
because it is the only option once the model becomes non-linear (Lecture 3)
or the dataset no longer fits in memory.

## 2.2 The update rule

Gradient descent repeats, for each parameter `θ`:

    θ ← θ − α * ∂L/∂θ

where `α` is the learning rate and `∂L/∂θ` is the partial derivative of the
loss with respect to that parameter. For MSE with a linear model the
derivative with respect to `w` is

    ∂L/∂w = −(2/n) * Σ xᵢ (yᵢ − f(xᵢ))

and with respect to `b` it is

    ∂L/∂b = −(2/n) * Σ (yᵢ − f(xᵢ))

Note what the gradient contains: the residual `(yᵢ − f(xᵢ))` multiplied by the
input `xᵢ`. A parameter that never appears alongside an error gets no signal.

## 2.3 The learning rate and its failure modes

The learning rate `α` controls step size and is the single most important knob.

- **Too small** — convergence is correct but painfully slow.
- **Too large** — the loss oscillates, then diverges to infinity.
- **About right** — loss decreases steadily and flattens out.

A practical recipe is to start around `0.01`–`0.1` for scaled features and
change it by factors of three, not by tiny increments. Plotting the loss per
iteration is not optional: a diverging curve is obvious within twenty steps.

## 2.4 Batch, stochastic and mini-batch

- **Batch gradient descent** computes the gradient over the whole training set
  before taking one step. Stable, but one step costs a full pass over the data.
- **Stochastic gradient descent (SGD)** takes a step after every single
  example. Noisy but very cheap per step; the noise can help escape shallow
  regions.
- **Mini-batch** is the compromise used in practice: a batch of 32–256 examples
  gives a gradient estimate good enough for a stable step and uses vectorised
  hardware efficiently.

The epoch is the unit of progress: one epoch means every training example has
been seen once, regardless of batch size.

## 2.5 Why feature scaling matters

If one feature ranges over `0–1` and another over `0–10000`, the loss surface
becomes a long narrow valley. A single global learning rate then has to be
small enough for the steep direction, which makes progress along the flat
direction glacial. Standardising features (subtract the mean, divide by the
standard deviation) turns the valley into a bowl and lets one learning rate
work for all parameters.

## 2.6 Convergence checks

Three standard stopping rules:

1. **Fixed budget** — stop after a set number of epochs.
2. **Gradient norm** — stop when `||∇L||` falls below a threshold.
3. **Validation patience** — stop when validation loss has not improved for
   `k` consecutive epochs (early stopping), and restore the best checkpoint.

Early stopping is the default choice in practice because it also acts as a
mild regulariser.

## 2.7 Common bug list

- Forgetting to zero the accumulated gradients between steps (loss explodes).
- Updating parameters before computing all gradients of the batch.
- Applying the `−(2/n)` factor twice, once in the loss and once in the update.
- Reporting the training loss as if it were the test score.
