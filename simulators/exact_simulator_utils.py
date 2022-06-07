from jax import numpy as jnp
from jax.lax import scan

from typing import Union


# TODO: add a description of what is below
sigma = jnp.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=jnp.complex128)
x = jnp.array([[0, 0, 1],
               [2 * jnp.sqrt(2) / 3, 0, -1 / 3],
               [-jnp.sqrt(2) / 3, jnp.sqrt(2 / 3), -1 / 3],
               [-jnp.sqrt(2) / 3, -jnp.sqrt(2 / 3), -1 / 3]], dtype=jnp.complex128)
M = jnp.eye(2) / 2 + jnp.tensordot(x, sigma, axes=1) / 2
complete_system, _, _ = jnp.linalg.svd(M)
complete_system = complete_system[..., 0]
M = M.reshape((4, 4))
M = M.T
M_inv = jnp.linalg.inv(M)
M_inv = M_inv.reshape((4, 2, 2))


def _apply_layer(gates_layer: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    """[This function apply a layer of gates to an input state. Number of qubits
    must be odd, while the number of gates in a layer must be even.]

    Args:
        gates_layer (complex valued jnp.ndarray of shape (n-1, 4, 4)): [set of
            n-1 two-qubit gates]
        state (complex valued jnp.ndarray of shape (2 ** n,)): [input state]

    Returns:
        complex valued jnp.ndarray of shape (2 ** n,): [output state]
    """

    def iter(state, gate):
        state = state.reshape((4, -1))
        state = jnp.tensordot(state, gate, axes=[[0], [1]])
        state = state.reshape((-1,))
        return state, None
    state, _ = scan(iter, state, gates_layer[::2])
    state = state.reshape((4, -1)).T.reshape((-1,))
    state, _ = scan(iter, state, gates_layer[1::2])
    return state


def _get_local_rho(state: jnp.ndarray, n: Union[jnp.ndarray, int]) -> jnp.ndarray:
    """[This function returns individual density matricies of all subsystems.]

    Args:
        state (complex valued jnp.ndarray of shape (2 ** n,)): [input state]
        n (Union[jnp.ndarray, int]): [number of subsystems]

    Returns:
        complex valued jnp.ndarray of shape (n, 2, 2): [density matrices of all subsystems]
    """

    def iter(state, xs):
        state = state.reshape((2, -1))
        rho = state @ state.conj().T
        state = state.T
        state = state.reshape((-1,))
        return state, rho

    _, rhos = scan(iter, state, None, length=n)
    return rhos


def _get_two_local_rho(state: jnp.ndarray, n: Union[jnp.ndarray, int]) -> jnp.ndarray:
    """[This function returns individual density matricies of all subsystems.]
    Args:
        state (complex valued jnp.ndarray of shape (2 ** n,)): [input state]
        n (Union[jnp.ndarray, int]): [number of subsystems]
    Returns:
        complex valued jnp.ndarray of shape (n, 2, 2): [density matrices of all subsystems]
    """

    def iter(state, xs):
        state = state.reshape((2, 2, -1))
        two_rho = jnp.tensordot(state, state.conj(), axes=((-1), (-1)))
        state = state.reshape((2, -1))
        state = state.T
        state = state.reshape((-1,))
        return state, two_rho

    _, rhos = scan(iter, state, None, length=n-1)
    return rhos


def _apply_control_signal(state: jnp.ndarray, u: jnp.ndarray, n: Union[int, jnp.ndarray]) -> jnp.ndarray:
    """[This function applies a control gate to the input state]

    Args:
        state (complex valued jnp.ndarray of shape (2 ** n,)): [input state]
        u (complex valued jnp.ndarray of shape (2, 2)): [unitary control gate]
        n (Union[int, jnp.ndarray]): [number of a subsystem that is under control gate]

    Returns:
        complex valued jnp.ndarray of shape (2 ** n,): [output state after control]
    """

    state = state.reshape((2 ** n, 2, -1))
    state = jnp.tensordot(u, state, axes=[[1], [1]])
    state = state.transpose((1, 0, 2))
    state = state.reshape((-1,))
    return state
