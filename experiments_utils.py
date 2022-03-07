from jax.scipy.special import xlogy
from jax.nn import relu
import jax.numpy as jnp
from functools import reduce
from typing import List

from simulators.control import random_isometric
from simulators.exact_simulator_utils import sigma


def entropy(rho: jnp.ndarray) -> jnp.ndarray:
    spec = jnp.linalg.svd(rho, compute_uv=False)
    spec = relu(spec)
    return -xlogy(spec, spec).sum(-1).real


def mutual_information(phis: jnp.ndarray) -> jnp.ndarray:
    rho1 = jnp.trace(phis / 2, axis1=-3, axis2=-4)
    rho2 = jnp.trace(phis / 2, axis1=-1, axis2=-2)
    s1 = entropy(rho1)
    s2 = entropy(rho2)
    phis = jnp.swapaxes(phis, -3, -2)
    phis = phis.reshape((*phis.shape[:-4], 4, 4))
    s12 = entropy(phis / 2)
    return s1 + s2 - s12


def rom2exact_init_state_converter(ro_env_state: List[jnp.ndarray]) -> jnp.ndarray:
    return reduce(lambda x, y: jnp.kron(x, y), ro_env_state)


def zero_control_seq(N: int) -> jnp.ndarray:
    return jnp.tile(jnp.eye(2, dtype=jnp.complex64)[jnp.newaxis], (N, 1, 1))


def random_control_seq(key: jnp.ndarray, N: int) -> jnp.ndarray:
    return random_isometric(key, (N, 2, 2))


def channels2rhos(quantum_channels: jnp.ndarray, init_system_state: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum('ijklmn,m,n->ijkl', quantum_channels, init_system_state, init_system_state.conj())

def rho2bloch(rho: jnp.ndarray) -> jnp.ndarray:
    return jnp.tensordot(rho, sigma, axes=[[1, 2], [2, 1]]).real

def controll_padding(
    control_gates: jnp.ndarray,
    startN: int,
    stopN: int,
    N: int,
    ) -> jnp.ndarray:
    prefix = jnp.eye(2)[jnp.newaxis]
    prefix = jnp.tile(prefix, (startN, 1, 1))
    postfix = jnp.eye(2)[jnp.newaxis]
    postfix = jnp.tile(postfix, (N - stopN, 1, 1))
    return jnp.concatenate([prefix, control_gates, postfix], axis=0)
