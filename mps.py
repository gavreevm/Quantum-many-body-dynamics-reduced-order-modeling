import jax.numpy as jnp
from typing import List, Tuple, Union
from functools import reduce
from mps_utils import _push_r_backward, _push_r_forward

# mps is a list with complex valued jnp.ndarray of shape (left_bond, dim, right_bond),
# where dim is a local dimension, left and right bonds take values 1, 2, ...

mps = List[jnp.ndarray]  # mps dtype (list with complex valued 3-rank tensors)

def set_to_forward_canonical(inp_mps: mps) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function sets mps to the forward (left) canonical form.
    It acts inplace in order to save memory.

    Args:
        inp_mps (mps): [input mps]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [logarithmic frobenius norm of the inp_mps and final r matrix]
    """

    # TODO: consider other construction then for loop here
    r = jnp.eye(inp_mps[0].shape[0])
    lognorm = 0.
    for i, ker in enumerate(inp_mps):
        ker, r = _push_r_backward(ker, r)
        norm = jnp.linalg.norm(r)
        r /= norm
        lognorm += jnp.log(norm)
        inp_mps[i] = ker
    return lognorm, r


def set_to_backward_canonical(inp_mps: mps) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function sets mps to the backward (right) canonical form.
    It acts inplace in order to save memory.

    Args:
        inp_mps (mps): [input mps]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [logarithmic frobenius norm of the inp_mps and final r matrix]
    """

    # TODO: consider other construction then for loop here
    r = jnp.eye(inp_mps[-1].shape[-1])
    lognorm = 0.
    for i, ker in enumerate(reversed(inp_mps)):
        ker, r = _push_r_forward(ker, r)
        norm = jnp.linalg.norm(r)
        r /= norm
        lognorm += jnp.log(norm)
        inp_mps[len(inp_mps) - i - 1] = ker
    return lognorm, r


def dot_prod(inp_mps1: mps, inp_mps2: mps) -> jnp.ndarray:
    """This function calculates the dot product of two mps
    and return logarithm of the result.

    Args:
        inp_mps1 (mps): [first input mps]
        inp_mps2 (mps): [sceond input mps]

    Returns:
        jnp.ndarray: [logorithm of the dot product]
    """

    def iter(carry, kers):
        lognorm, state = carry
        ker1, ker2 = kers
        ker2 = ker2.conj()
        state = jnp.tensordot(ker1, state, axes=1)
        state = jnp.tensordot(state, ker2, axes=[[1, 2], [1, 2]])
        norm = jnp.trace(state)
        state /= norm
        lognorm += jnp.log(norm)
        return lognorm, state

    lognorm, _ = reduce(iter, zip(reversed(inp_mps1), reversed(inp_mps2)), (0., jnp.eye(inp_mps1[-1].shape[-1])))

    return lognorm

def truncate_forwaed_canonical(inp_mps: mps, eps: Union[float, jnp.ndarray]) -> None:
    pass