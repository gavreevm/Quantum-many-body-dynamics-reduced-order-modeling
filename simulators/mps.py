"""
MPS module
mps is a list with complex valued jnp.ndarray of shape (left_bond, dim, right_bond),
where dim is a local dimension, left and right bonds take values 1, 2, ...
                               1
                               |
MPO indices enumearion:   0 -- O -- 2, MPS indices enumeration:  0 -- O -- 2.
                               |                                      |
                               3                                      1
"""

from typing import List, Union
from functools import reduce
import jax.numpy as jnp
from simulators.mps_utils import (_push_r_backward,
                                  _push_r_forward,
                                  _push_orth_center_forward,
                                  _set_rank)


mps = List[jnp.ndarray]  # mps dtype (list with complex valued 3-rank tensors)
mpo = List[jnp.ndarray]  # mpo dtype (list with complex valued 4-rank tensors)


def set_to_forward_canonical(inp_mps: mps) -> jnp.ndarray:
    """This function sets mps to the forward (left) canonical form.
    It acts inplace in order to save memory.
    Args:
        inp_mps (mps): [input mps]
    Returns:
        jnp.ndarray: [logarithmic frobenius norm of the inp_mps]
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
    inp_mps[-1] = jnp.tensordot(inp_mps[-1], r, axes=1)
    return lognorm


def set_to_backward_canonical(inp_mps: mps) -> jnp.ndarray:
    """This function sets mps to the backward (right) canonical form.
    It acts inplace in order to save memory.

    Args:
        inp_mps (mps): [input mps]

    Returns:
        jnp.ndarray: [logarithmic frobenius norm of the inp_mps]
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
    inp_mps[0] = jnp.tensordot(r, inp_mps[0], axes=1)
    return lognorm


def dot_prod(inp_mps1: mps, inp_mps2: mps, use_conj: bool = True) -> jnp.ndarray:
    """This function calculates the dot product of two mps
    and return logarithm of the result.

    Args:
        inp_mps1 (mps): [first input mps]
        inp_mps2 (mps): [sceond input mps]
        use_conj (bool): [flag showing whether to use complex conjugate of the second
            argument or not]

    Returns:
        jnp.ndarray: [logorithm of the dot product]
    """

    def iter(carry, kers):
        lognorm, state = carry
        ker1, ker2 = kers
        if use_conj:
            ker2 = ker2.conj()
        state = jnp.tensordot(ker1, state, axes=1)
        state = jnp.tensordot(state, ker2, axes=[[1, 2], [1, 2]])
        norm = jnp.trace(state)
        state /= norm
        lognorm += jnp.log(norm)
        return lognorm, state

    lognorm, _ = reduce(iter, zip(reversed(inp_mps1), reversed(inp_mps2)), (
                                0., jnp.eye(inp_mps1[-1].shape[-1])))

    return lognorm


def truncate_forward_canonical(inp_mps: mps, eps: Union[float, jnp.ndarray]) -> None:
    """This function contructs forward canonical form of a mps. It acts inplace
    in order to save memory.
    Args:
        inp_mps (mps): [input mps]
        eps (Union[float, jnp.ndarray]): [accuracy of truncation]
    """

    u = jnp.eye(inp_mps[-1].shape[-1])
    spec = jnp.ones((inp_mps[-1].shape[-1],))
    for i, ker in enumerate(reversed(inp_mps)):
        ker, u, spec = _push_orth_center_forward(ker, u, spec, eps)
        inp_mps[len(inp_mps) - i - 1] = ker
    return u


def truncate_very_last_edge_backward_canonical(inp_mps: mps,
                                               eps: Union[float, jnp.ndarray]) -> None:
    """This function truncates the last edge of a mps in the backward (right) canonical form.
    It is necessary to perform complite truncation of an environment. The function
    acts inplace in order to save memory.
    Args:
        inp_mps (mps): [input mps]
        eps (Union[float, jnp.ndarray]): [accuracy]
    """

    _, dim, right_bond = inp_mps[0].shape
    dens = jnp.tensordot(inp_mps[0].conj(), inp_mps[0], axes=[[0], [0]])
    dens = dens.reshape((-1, dens.shape[1] * dens.shape[2]))
    _, s, vh = jnp.linalg.svd(dens, full_matrices=False)
    eta = _set_rank(s, eps)
    s, vh = s[:eta], vh[:eta, :]
    vh = vh.reshape((-1, dim, right_bond))
    inp_mps[0] = jnp.sqrt(s)[:, jnp.newaxis, jnp.newaxis] * vh


def mpo_mps_product(inp_mpo: mpo, inp_mps: mps, reverse: bool = False) -> None:
    """This function compute mpo mps product (tensorized matvec).
    It acts inplace updating mps kernels in order to save memory.

    Args:
        inp_mpo (mpo): [input mp0]
        inp_mps (mps): [input mps]
        reverse (bool): [flag showing whether to use "matves" of "vecmat (reverse)"]
    """

    for i, (mpo_ker, mps_ker) in enumerate(zip(inp_mpo, inp_mps)):
        mpo_left_bond, _, mpo_right_bond, _ = mpo_ker.shape
        mps_left_bond, _, mps_right_bond = mps_ker.shape
        if reverse:
            inp_mps[i] = jnp.einsum('ilkj,mjn->imlkn', mpo_ker, mps_ker).reshape((
                                                    mpo_left_bond * mps_left_bond, -1,
                                                    mpo_right_bond * mps_right_bond))
        else:
            inp_mps[i] = jnp.einsum('ijkl,mjn->milnk', mpo_ker, mps_ker).reshape((
                                                    mpo_left_bond * mps_left_bond, -1,
                                                    mpo_right_bond * mps_right_bond))
