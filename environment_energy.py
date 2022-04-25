"""
Module for decoding reduced order environment state
####################################################################################################
Allows:
    - Evironment state decoding
    - Computing composite (System + Environment) energy from reduced order environment state
"""

from functools import reduce
import jax.numpy as jnp
from jax import jit

from simulators.exact_simulator_utils import sigma
from simulators.mps import mpo


def statevector_from_ro_state(ro_state, isometries):
    """ Returns the decoded system-environment state from reduced state
        Args:
            ro_state: resulting embedding vector
            isometries: isometric matrices from truncation procedure
        Returns: system+environment statevector """
    @jit
    def contract(state, isometry):
        top_iso, bot_iso = isometry
        # reshape state to extract physical dimensions
        state = state.reshape((top_iso.shape[1], -1) + state.shape[1:])
        state = state.reshape(state.shape[:-1] + (-1, bot_iso.shape[1]))
        # contract with isometries
        state = jnp.tensordot(top_iso, state, axes=((1), (0)))
        state = jnp.tensordot(state, bot_iso, axes=((-1), (1)))
        return state

    top, bot = isometries
    return reduce(contract, zip(top, bot), ro_state).reshape(-1)



def params2hamiltonian_mpo(couplings: jnp.ndarray,
                           fields: jnp.ndarray) -> mpo:
    """ Transform local hamiltonian to MPO form
    Args: couplings (jnp.ndarray) coupling parameters (site, {x, y, z})
          fields (jnp.ndarray) local field parameters (site, {x, y, z})
    Returns:
          Hamiltonian in mpo form (mpo)
    """

    local_parts = jnp.tensordot(fields, sigma, axes=((1), (0)))
    left_vec = jnp.array([local_parts[0]] + \
                         list((couplings[0] * sigma.T).T) + \
                         [jnp.eye(2)])
    right_vec = jnp.array([jnp.eye(2)] + list(sigma) + [local_parts[-1]])
    zeros = 4 * [jnp.zeros((2, 2), dtype=jnp.complex64)]
    const_rows = jnp.array([[jnp.eye(2)] + zeros, [sigma[0]] + zeros,
                              [sigma[1]] + zeros, [sigma[2]] + zeros])
    mid_blocks = []
    for i in range(1, len(fields) - 1):
        site_row = jnp.array([local_parts[i]] + \
                   list((couplings[i] * sigma.T).T) + \
                   [jnp.eye(2)])
        block = jnp.concatenate([const_rows, site_row[jnp.newaxis, :]], axis=0)
        mid_blocks.append(block)

    return [left_vec] + mid_blocks + [right_vec]


