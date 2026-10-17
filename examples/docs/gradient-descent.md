# Gradient descent

Gradient descent minimises a loss function by repeatedly stepping in the direction of the negative gradient. The learning rate controls the step size: too large and training diverges, too small and it converges slowly.

Stochastic gradient descent estimates the gradient on mini-batches. Adam keeps running averages of gradients and squared gradients to adapt the learning rate per parameter. Regularisation techniques such as weight decay and dropout reduce overfitting.
