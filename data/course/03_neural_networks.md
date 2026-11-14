# Lecture 3 — Neural Networks and Backpropagation

Course: Introduction to Machine Learning (demo material for VARK Check).

## 3.1 Why a linear model is not enough

A linear model can only draw a straight boundary. The classic counter-example
is XOR: the four points `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels
`0, 1, 1, 0` cannot be separated by any single line. Adding a hidden layer and
a non-linear activation lets the network build a curved boundary by composing
simple ones.

## 3.2 Layers and activations

A one-hidden-layer network is

    h = σ(W₁ x + b₁)
    ŷ = W₂ h + b₂

`σ` is an element-wise non-linearity. Without it, stacking layers collapses
back into a single linear map, because a composition of affine functions is
still affine. Common choices:

- **ReLU** — `max(0, z)`. Cheap, does not saturate for positive inputs, the
  default for hidden layers.
- **Sigmoid** — `1 / (1 + e^(−z))`. Squashes to `(0,1)`; used for binary
  outputs, but saturates badly in hidden layers.
- **Tanh** — zero-centred sigmoid, same saturation problem.
- **Softmax** — turns a vector into a probability distribution; used at the
  output for multi-class problems.

## 3.3 The forward pass

The forward pass evaluates the network layer by layer and caches every
intermediate value, because the backward pass will need them. For each layer
store the input, the pre-activation `z`, and the activation `a`. Memory for the
forward cache is why training needs far more GPU memory than inference.

## 3.4 Loss and the chain rule

Training still minimises a loss, usually cross-entropy for classification:

    L = −Σ yₖ log(ŷₖ)

The gradient of the loss with respect to a weight deep in the network is
computed with the chain rule: the derivative of the loss with respect to the
output, multiplied by the derivative of each function on the path back to that
weight. Backpropagation is nothing more than a systematic, efficient bookkeeping
scheme for those products — it computes all parameter gradients in one backward
sweep, at roughly the cost of the forward pass.

## 3.5 The backpropagation algorithm

1. Run the forward pass and cache the activations.
2. Compute the gradient of the loss at the output, `δ_out`.
3. Propagate backwards: for each layer, `δ = (Wᵀ δ_next) ⊙ σ'(z)`.
4. Accumulate the parameter gradients: `∂L/∂W = δ a_prevᵀ`, `∂L/∂b = δ`.
5. Average the gradients over the mini-batch.
6. Apply the gradient-descent update `θ ← θ − α · ∂L/∂θ`.

The `⊙` is the element-wise product; the transposed weight matrix routes the
error back to the previous layer's neurons.

## 3.6 Vanishing and exploding gradients

Sigmoid and tanh squash their input; their derivative is at most `0.25` for
sigmoid. Multiplying many such factors while going backwards drives the
gradient towards zero, so early layers stop learning. ReLU activations,
He/Xavier initialisation, batch normalisation and residual connections all
address this. The opposite failure, exploding gradients, is usually handled by
gradient clipping: rescale the gradient whenever its norm exceeds a threshold.

## 3.7 Regularisation

- **Dropout** — randomly zero a fraction of activations during training, which
  stops neurons from co-adapting. Disabled at inference time.
- **Weight decay** — add `λ ||W||²` to the loss to keep weights small.
- **Early stopping** — the cheapest regulariser, using a validation split.
- **Data augmentation** — more effective data, cheaply.

## 3.8 Debugging checklist

A network that does not learn is almost always one of: learning rate too large
(loss is NaN), labels misaligned with inputs, no non-linearity between layers,
or an unnormalised input. Sanity check first: can the model overfit a single
mini-batch to near-zero loss? If it cannot, the bug is in the pipeline, not in
the optimiser.
