import jax.numpy as jnp
from jax import scipy
from typing import Dict
from simulators.exact_simulator_utils import sigma

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
