from trax.fastmath import numpy as jnp
from trax.optimizers import base as opt_base


class SGD(opt_base.Optimizer):
    """Stochastic gradient descent (SGD) optimizer."""

    def init(self, weights):
        return None

    def update(self, step, grads, weights, slots, opt_params):
        del step, slots
        lr = opt_params["learning_rate"]
        new_weights = jnp.subtract(
            weights, jnp.multiply(lr, grads).astype(weights.dtype)
        )

        return new_weights, None
