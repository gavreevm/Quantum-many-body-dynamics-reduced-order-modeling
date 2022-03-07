from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import pytest
import jax.numpy as jnp
from jax import random

from simulators.exact_simulator_utils import _apply_layer, _get_local_rho, _apply_control_signal

key = random.PRNGKey(42)
key, subkey, subsubkey = random.split(key, 3)

random_gates_layer  = random.normal(subkey, (6, 4, 4, 2))
random_gates_layer  = random_gates_layer [..., 0] + 1j * random_gates_layer [..., 1]
random_gates_layer , _ = jnp.linalg.qr(random_gates_layer)

u = random.normal(subsubkey, (2, 2, 2))
u = u[..., 0] + 1j * u[..., 1]
u, _ = jnp.linalg.qr(u)

state = random.normal(key, (2 ** 7, 2))
state = state[..., 0] + 1j * state[..., 1]


@pytest.mark.parametrize("gates,state", [(random_gates_layer, state)])
def test_apply_layer(gates, state):
    """Apply random gates layer to a random state and check the correctness."""
    smart_state = _apply_layer(gates, state)
    u1 = jnp.tensordot(jnp.tensordot(jnp.tensordot(gates[0], gates[2], axes=0), gates[4], axes=0), jnp.eye(2), axes=0)
    u1 = u1.transpose((0, 2, 4, 6, 1, 3, 5, 7))
    u1 = u1.reshape((2 ** 7, 2 ** 7))
    u2 = jnp.tensordot(jnp.eye(2), jnp.tensordot(jnp.tensordot(gates[1], gates[3], axes=0), gates[5], axes=0), axes=0)
    u2 = u2.transpose((0, 2, 4, 6, 1, 3, 5, 7))
    u2 = u2.reshape((2 ** 7, 2 ** 7))
    naive_state = jnp.tensordot(u2, jnp.tensordot(u1, state, axes=1), axes=1)
    err = jnp.linalg.norm(smart_state - naive_state)
    assert err < 1e-10, "Distance between 'naive' and 'smart' final states is too big"


@pytest.mark.parametrize("state", [state])
def test_get_local_rho(state):
    local_rhos = _get_local_rho(state, 7)
    big_rho = jnp.tensordot(state, state.conj(), axes=0)
    big_rho = big_rho.reshape(14 * (2,))
    for i in range(7):
        rho = jnp.einsum('ijkimk->jm', big_rho.reshape((2 ** i, 2, 2 ** (6 - i), 2 ** i, 2, 2 ** (6 - i))))
        err = jnp.linalg.norm(rho - local_rhos[i])
        assert err < 1e-10, "{}-th density matrix is incorrect".format(i)

@pytest.mark.parametrize("state,u,n", [(state, u, 3)])
def test_apply_control_signal(state, u, n):
    control_state = _apply_control_signal(state, u, n)
    local_rho = _get_local_rho(state, 7)[n]
    control_local_rho = _get_local_rho(control_state, 7)[n]
    err = jnp.linalg.norm(control_local_rho - u @ local_rho @ u.conj().T)
    assert err < 1e-10, "State after control is incorrect"
