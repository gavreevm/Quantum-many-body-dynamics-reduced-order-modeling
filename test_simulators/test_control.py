from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import pytest
import jax.numpy as jnp
from jax import random

from simulators.control import optimize, random_isometric

def test_control():
    key = random.PRNGKey(42)
    subkey1, subkey2 = random.split(key)
    a = random.normal(subkey1, (10, 10, 2))
    a = a[..., 0] + 1j * a[..., 1]

    constant_params = [a]
    control_signal = random_isometric(subkey2, (2, 10, 10))
    number_of_epoch = 2
    epoch_size = 1000
    learning_rate = 0.5

    def loss_fn(constant_params, control_signal):
        return -(jnp.abs(jnp.diag(control_signal[0] @ constant_params[0] @ control_signal[1])) ** 2).sum()

    control_signal, hist = optimize(loss_fn, constant_params, control_signal, number_of_epoch, epoch_size, learning_rate)

    opt_based_lmbd = jnp.sort(jnp.abs(jnp.diag(control_signal[0] @ a @ control_signal[1])))[::-1]
    _, exact_lmbd, _ = jnp.linalg.svd(a)

    assert jnp.linalg.norm(exact_lmbd - opt_based_lmbd) < 1e-10, "Test optimization problem is solved incorrectly"
