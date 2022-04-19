""" MPS utils module
"""

import jax.numpy as jnp
from typing import Union, Tuple


def _rev_qr(a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function returns "reversed" qr decomposition.
    Args:
        a (complex valued jnp.ndarray of shape (m, n)): [input matrix]
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [matrices q and r]
    """

    q, r = jnp.linalg.qr(a.T)
    q, r = q.T, r.T
    return q, r


def _set_rank(spec: jnp.ndarray, eps: Union[jnp.ndarray, float]) -> jnp.ndarray:
    """This function returns the minimal number of singular values (rank) that are necessary
    to keep, in order to guarantee the given level of accuracy epsilon.
    Args:
        spec (real valued jnp.ndarray of shape (n,)): [set of singular values]
        eps (Union[jnp.ndarray, float]): [accuracy]

    Returns:
        int valued jnp.ndarray of shape (1,): [rank]
    """

    cum_sq_sum = jnp.cumsum(spec[::-1] ** 2)
    sq_sum = (spec ** 2).sum()
    trsh = (jnp.sqrt(cum_sq_sum / sq_sum) > eps).sum()
    return trsh


def _push_r_backward(ker: jnp.ndarray, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function is a part of the subroutine of setting mps in the forward (left)
    canonical form. It pushes r (a matrix from the qr decomposition) backward in time.
    Args:
        ker (complex valued jnp.ndarray of shape (r_left, d, r_right)): [current kernel]
        r (complex valued jnp.ndarray of shape (-1, r_left)): [previous r matrix]
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [new kernel and r matrix]
    """

    _, dim, right_bond = ker.shape
    left_bond = r.shape[0]
    ker = jnp.tensordot(r, ker, axes=1)
    ker = ker.reshape((-1, right_bond))
    ker, r = jnp.linalg.qr(ker)
    ker = ker.reshape((left_bond, dim, -1))
    return ker, r


def _push_r_forward(ker: jnp.ndarray, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function is a part of the subroutine of setting mps in the backward (right)
       canonical form. It pushes r (a matrix from the rq decomposition) forward in time.
    Args:
        ker (complex valued jnp.ndarray of shape (r_left, d, r_right)): [current kernel]
        r (complex valued jnp.ndarray of shape (r_right, -1)): [previous r matrix]
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [new kernel and r matrix]
    """

    left_bond, dim, _ = ker.shape
    right_bond = r.shape[1]
    ker = jnp.tensordot(ker, r, axes=1)
    ker = ker.reshape((left_bond, -1))
    ker, r = _rev_qr(ker)
    ker = ker.reshape((-1, dim, right_bond))
    return ker, r


#TODO: add push orth. center backward
def _push_orth_center_forward(ker: jnp.ndarray,
                              u: jnp.ndarray,
                              spec: jnp.ndarray,
                              eps: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function moves the orthogonality center forward in time and also performs
    truncation of singular values spectrum. It is part of mps truncation subroutine.
    Args:
        ker (complex valued jnp.ndarray of shape (r_left, d, r_right)): [current kernel]
        u (complex valued jnp.ndarray of shape (r_right, -1)): [u from usv^dag from the previous step]
        spec (complex valued jnp.ndarray of shape (-1,)): [signular values from the previous step]
        eps (Union[float, jnp.ndarray]): [accuracy of truncation]
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: [truncated kernel, new u matrix and new spec]"""

    ker = jnp.tensordot(ker, u, axes=1)
    q = ker * spec
    right_bond = q.shape[0]
    q = q.reshape((right_bond, -1))
    u, s, _ = jnp.linalg.svd(q, full_matrices=False)
    trsh = _set_rank(s, eps)
    u, s = u[:, :trsh], s[:trsh]
    ker = jnp.tensordot(u.T.conj(), ker, axes=1)
    return ker, u, s
