# Lecture 1 — Linear Regression and the Idea of a Loss Function

Course: Introduction to Machine Learning (demo material for VARK Check).
This file is intentionally written as real lecture notes so the agent has
something to ground its answers in. Every generated artefact must cite a
section from this folder.

## 1.1 Supervised learning in one paragraph

In supervised learning we are given a dataset of pairs `(x, y)` and we want a
function `f` such that `f(x)` is close to `y` for new, unseen `x`. The dataset is
split into a training set (used to fit the parameters) and a held-out set (used
to estimate how well the model generalises). Nothing else changes: every
supervised method is a choice of *model family* plus a choice of *loss*.

## 1.2 The linear model

The simplest model family assumes the output is an affine function of the
inputs:

    f(x) = w · x + b

where `w` is a vector of weights and `b` is a bias (intercept). With one input
feature this is the familiar line `f(x) = w x + b`. The weights are the
*parameters*; the inputs are *features*. A linear model can only represent
straight lines, planes and hyperplanes — that limitation is exactly what
Lecture 3 fixes with hidden layers.

## 1.3 Mean squared error

Given `n` training examples, the mean squared error is

    MSE(w, b) = (1/n) * Σ (yᵢ - (w · xᵢ + b))²

The residual `yᵢ - f(xᵢ)` is the signed error on example `i`. Squaring does two
things: it makes all residuals positive, and it punishes large residuals much
more than small ones. The factor `1/n` keeps the number comparable across
dataset sizes.

## 1.4 Why squared error and not absolute error

MSE is convenient because it is smooth and differentiable everywhere, which
makes the gradient well defined (Lecture 2 depends on this). Mean absolute
error is also legitimate and is more robust to outliers, but its derivative is
undefined at zero and constant elsewhere, so it needs different optimisation
machinery (subgradients). If the noise in `y` is assumed Gaussian with constant
variance, minimising MSE is equivalent to maximum likelihood estimation.

## 1.5 Training, validation and test sets

Three splits, three jobs:

- **Training set** — used to choose the parameters `w` and `b`.
- **Validation set** — used to choose hyperparameters (learning rate, model
  size, regularisation strength).
- **Test set** — touched once, at the end, to report a number you can defend.

Using the test set to make decisions turns it into a validation set and the
reported score becomes optimistic. This is one of the most common mistakes in
student projects.

## 1.6 Overfitting and underfitting

Underfitting means the model is too simple to capture the pattern: both
training and validation error are high. Overfitting means the model has
memorised the training set: training error is low while validation error is
high. The gap between the two curves is the diagnostic. Regularisation
(penalising large weights) and more data both shrink that gap.

## 1.7 Worked example

Suppose we have three examples `(1, 2)`, `(2, 4)`, `(3, 6)` and the model
`f(x) = 2x`. Residuals are `0, 0, 0`, so MSE is `0`. For the model
`f(x) = 1x + 0.5` residuals are `0.5, 1.5, 2.5`; squared residuals are
`0.25, 2.25, 6.25`; MSE is `8.75 / 3 ≈ 2.92`. The first model is clearly
better, and the loss number says so.
