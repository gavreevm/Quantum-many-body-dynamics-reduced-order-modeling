import jax.numpy as jnp
from jax import scipy
from typing import Dict
from simulators.exact_simulator_utils import sigma
from jax import random, vmap

def params2gates_layer(params: Dict) -> jnp.ndarray:
    """[This function returns a gate layer from dict
    with model parameters]

    Args:
        params (Dict): [model parameters]

    Returns:
        complex valued jnp.ndarray of shape (n-1, 4, 4): [gates]
    """

    Jx, Jy, Jz = params['Jx'], params['Jy'], params['Jz']
    hx, hy, hz = params['hx'], params['hy'], params['hz']
    tau = params['tau']
    n = params['n']

    H_central = (Jx * jnp.tensordot(sigma[0], sigma[0], axes=0)
                + Jy * jnp.tensordot(sigma[1], sigma[1], axes=0)
                + Jz * jnp.tensordot(sigma[2], sigma[2], axes=0)
                + (hz / 2) * jnp.tensordot(sigma[2], jnp.eye(2), axes=0)
                + (hz / 2) * jnp.tensordot(jnp.eye(2), sigma[2], axes=0)
                + (hy / 2) * jnp.tensordot(sigma[1], jnp.eye(2), axes=0)
                + (hy / 2) * jnp.tensordot(jnp.eye(2), sigma[1], axes=0)
                + (hx / 2) * jnp.tensordot(sigma[0], jnp.eye(2), axes=0)
                + (hx / 2) * jnp.tensordot(jnp.eye(2), sigma[0], axes=0))
    H_central = H_central.transpose((0, 2, 1, 3))
    H_central = H_central.reshape((4, 4))
    U_central = scipy.linalg.expm(-1j * H_central * tau)

    H_up = (Jx * jnp.tensordot(sigma[0], sigma[0], axes=0)
            + Jy * jnp.tensordot(sigma[1], sigma[1], axes=0)
            + Jz * jnp.tensordot(sigma[2], sigma[2], axes=0)
            + hz * jnp.tensordot(sigma[2], jnp.eye(2), axes=0)
            + (hz / 2) * jnp.tensordot(jnp.eye(2), sigma[2], axes=0)
            + hy * jnp.tensordot(sigma[1], jnp.eye(2), axes=0)
            + (hy / 2) * jnp.tensordot(jnp.eye(2), sigma[1], axes=0)
            + hx * jnp.tensordot(sigma[0], jnp.eye(2), axes=0)
            + (hx / 2) * jnp.tensordot(jnp.eye(2), sigma[0], axes=0))
    H_up = H_up.transpose((0, 2, 1, 3))
    H_up = H_up.reshape((4, 4))
    U_up = scipy.linalg.expm(-1j * H_up * tau)

    H_down = (Jx * jnp.tensordot(sigma[0], sigma[0], axes=0)
            + Jy * jnp.tensordot(sigma[1], sigma[1], axes=0)
            + Jz * jnp.tensordot(sigma[2], sigma[2], axes=0)
            + (hz / 2) * jnp.tensordot(sigma[2], jnp.eye(2), axes=0)
            + hz * jnp.tensordot(jnp.eye(2), sigma[2], axes=0)
            + (hy / 2) * jnp.tensordot(sigma[1], jnp.eye(2), axes=0)
            + hy * jnp.tensordot(jnp.eye(2), sigma[1], axes=0)
            + (hx / 2) * jnp.tensordot(sigma[0], jnp.eye(2), axes=0)
            + hx * jnp.tensordot(jnp.eye(2), sigma[0], axes=0))
    H_down = H_down.transpose((0, 2, 1, 3))
    H_down = H_down.reshape((4, 4))
    U_down = scipy.linalg.expm(-1j * H_down * tau)

    U_down = U_down[jnp.newaxis]
    U_up = U_up[jnp.newaxis]
    U_central = jnp.tile(U_central[jnp.newaxis], (n-3, 1, 1))

    return jnp.concatenate([U_up, U_central, U_down], axis=0)


def sample_disordered_floquet(params: Dict) -> jnp.ndarray:
    """[This function returns gate layer that corresponds to a
    particular implementation of disrodered Floquet protocol with
    TFI hamiltonian.]

    Args:
        params (Dict): [parameters of a model]

    Returns:
        complex valued jnp.ndarray of shape (n-1, 4, 4): [gates]
    """

    random_seed, n, hx, J = params['random_seed'], params['n'], params['hx'], params['J']
    key = random.PRNGKey(random_seed)
    hz = 2 * jnp.pi * random.uniform(key, (n,))
    uz = vmap(scipy.linalg.expm)(1j * jnp.tensordot(hz, sigma[2], axes=0))
    ux = scipy.linalg.expm(1j * hx * sigma[0])
    uzz = scipy.linalg.expm(1j * J * jnp.tensordot(sigma[2], sigma[2], axes=0).transpose((0, 2, 1, 3)).reshape((4, 4)))
    uzz = uzz.reshape((2, 2, 2, 2))
    second_layer = []
    for z in uz[1:].reshape(-1, 2, 2, 2):
        second_layer.append(jnp.einsum('il,jk,lq,kp,qpmn->ijmn', ux, ux, z[0], z[1], uzz).reshape((4, 4))[jnp.newaxis])
    second_layer = jnp.concatenate(second_layer, axis=0)[:, jnp.newaxis]
    first_gate = jnp.einsum('ik,kq,qjmn->ijmn', ux, uz[0], uzz).reshape((4, 4))[jnp.newaxis]
    first_layer = jnp.tile(uzz.reshape((4, 4))[jnp.newaxis], (n // 2 - 1, 1, 1))
    first_layer = jnp.concatenate([first_gate, first_layer], axis=0)[:, jnp.newaxis]
    gates = jnp.concatenate([first_layer, second_layer], axis=1)
    gates = gates.reshape((-1, 4, 4))
    return gates
