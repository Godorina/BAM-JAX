"""Shared loss functions for training and evaluation."""

import jax.numpy as jnp


def huber_loss(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss."""
    abs_x = jnp.abs(x)
    return jnp.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))


def mae_loss(x: jnp.ndarray, delta: float = None) -> jnp.ndarray:
    return jnp.abs(x)


def mse_loss(x: jnp.ndarray, delta: float = None) -> jnp.ndarray:
    return x ** 2


LOSS_FUNCTIONS = {
    'huber': huber_loss,
    'mae': mae_loss,
    'mse': mse_loss,
}
