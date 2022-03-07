from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import pytest

import jax.numpy as jnp
from jax import random
from simulators.mps_utils import _set_rank, _push_r_backward, _push_r_forward, _push_orth_center_forward

key = random.PRNGKey(42)

def test_set_rank():
    """Check the correctness of rank truncation."""

    spec1 = jnp.stack([jnp.ones((5,)), jnp.zeros((5,))])
    rank1 = _set_rank(spec1, 1e-8)
    spec2 = jnp.arange(0,-10, -1)
    spec2 = jnp.exp(0.1 * spec2)
    spec2 = spec2 / jnp.linalg.norm(spec2)
    rank2 = _set_rank(spec2, spec2[-1])
    assert rank1 == 5
    assert rank2 == 9


test_push_r_backward_input = [
    (key, ((1, 2, 3), (3, 4, 5))),
    (key, ((15, 2, 13), (13, 3, 17)))
]
@pytest.mark.parametrize("key,shapes", test_push_r_backward_input)
def test_push_r_backward(key, shapes):
    """Set small mps (2 nodes) to canonical form and then check."""

    shape1, shape2 = shapes
    subkey1, subkey2 = random.split(key)
    ker1 = random.normal(subkey1, (*shape1, 2))
    ker1 = ker1[..., 0] + 1j * ker1[..., 1]
    ker2 = random.normal(subkey2, (*shape2, 2))
    ker2 = ker2[..., 0] + 1j * ker2[..., 1]
    a = jnp.tensordot(ker1, ker2, axes=1)
    a = a.reshape((-1, shape2[-1]))
    r = jnp.eye(shape1[0], dtype=jnp.complex64)
    ker1, r = _push_r_backward(ker1, r)
    ker2, r = _push_r_backward(ker2, r)
    big_q = jnp.tensordot(ker1, ker2, axes=1)
    big_q = big_q.reshape((-1, r.shape[0]))
    is_isometric = jnp.linalg.norm(big_q.T.conj() @ big_q - jnp.eye(shape2[-1], dtype=jnp.complex64))
    is_triangular = jnp.linalg.norm(jnp.tril(r, -1))
    is_decomposition = jnp.linalg.norm(a - big_q @ r)
    assert is_isometric < 1e-10, "Resulting big_q is not an isometric matrix"
    assert is_triangular < 1e-10, "Resulting r is not a triu matrix"
    assert is_decomposition < 1e-10, "Resulting big_q @ r is not a decomposition of a"


test_push_r_forward_input = [
    (key, ((5, 4, 3), (3, 2, 1))),
    (key, ((17, 3, 13), (13, 2, 15)))
]
@pytest.mark.parametrize("key,shapes", test_push_r_forward_input)
def test_push_r_forward(key, shapes):
    """Set small mps (2 nodes) to canonical form and then check."""

    shape1, shape2 = shapes
    subkey1, subkey2 = random.split(key)
    ker1 = random.normal(subkey1, (*shape1, 2))
    ker1 = ker1[..., 0] + 1j * ker1[..., 1]
    ker2 = random.normal(subkey2, (*shape2, 2))
    ker2 = ker2[..., 0] + 1j * ker2[..., 1]
    a = jnp.tensordot(ker1, ker2, axes=1)
    a = a.reshape((shape1[0], -1))
    r = jnp.eye(shape2[-1], dtype=jnp.complex64)
    ker2, r = _push_r_forward(ker2, r)
    ker1, r = _push_r_forward(ker1, r)
    big_q = jnp.tensordot(ker1, ker2, axes=1)
    big_q = big_q.reshape((r.shape[1], -1))
    is_isometric = jnp.linalg.norm(big_q @ big_q.conj().T - jnp.eye(shape1[0], dtype=jnp.complex64))
    is_triangular = jnp.linalg.norm(jnp.triu(r, 1))
    is_decomposition = jnp.linalg.norm(a - r @ big_q)
    assert is_isometric < 1e-10, "Resulting big_q is not an isometric matrix"
    assert is_triangular < 1e-10, "Resulting r is not a tril matrix"
    assert is_decomposition < 1e-10, "Resulting r @ big_q is not a decomposition of a"


test_push_orth_center_forward = [
    (key, ((5, 3, 2), (2, 2, 3), (3, 4, 1)))
]
@pytest.mark.parametrize("key,shapes", test_push_orth_center_forward)
def test_push_orth_center_forward(key, shapes):
    """Truncate small mps (3 nodes) and check the result"""

    shape1, shape2, shape3 = shapes
    subkey1, subkey2, subkey3 = random.split(key, 3)
    # kernels are designed in such a way that their bond dimensions are greater
    # then rank
    ker1 = random.normal(subkey1, (*shape1, 2))
    ker1 = ker1[..., 0] + 1j * ker1[..., 1]
    ker1 = jnp.tensordot(ker1, jnp.ones((3,)), axes=0)
    ker1 = ker1.reshape((*shape1[:2], 3*shape1[-1]))

    ker2 = random.normal(subkey2, (*shape2, 2))
    ker2 = ker2[..., 0] + 1j * ker2[..., 1]
    ker2 = jnp.tensordot(ker2, jnp.eye(3), axes=0)
    ker2 = ker2.transpose((0, 3, 1, 2, 4))
    ker2 = ker2.reshape((3*shape2[0], shape2[1], 3*shape2[-1]))

    ker3 = random.normal(subkey3, (*shape3, 2))
    ker3 = ker3[..., 0] + 1j * ker3[..., 1]
    ker3 = jnp.tensordot(ker3, jnp.ones((3,)), axes=0)
    ker3 = ker3.transpose((0, 3, 1, 2))
    ker3 = ker3.reshape((3*shape3[0], *shape3[1:]))

    # set to forward canonical
    r = jnp.eye(ker1.shape[0])
    ker1, r = _push_r_backward(ker1, r)
    ker2, r = _push_r_backward(ker2, r)
    ker3, r = _push_r_backward(ker3, r)

    tensor_exact = jnp.tensordot(ker1, jnp.tensordot(ker2, ker3, axes=1), axes=1)
    half_way_tensor_exact = jnp.tensordot(ker2, ker3, axes=1)
    half_way_rho_exact = jnp.tensordot(half_way_tensor_exact.conj(), half_way_tensor_exact, axes=[[0], [0]])

    # truncate
    spec = r[0]
    u = jnp.eye(1)
    ker3, u, spec = _push_orth_center_forward(ker3, u, spec, 1e-10)
    ker2, u, spec = _push_orth_center_forward(ker2, u, spec, 1e-10)
    ker1, u, spec = _push_orth_center_forward(ker1, u, spec, 1e-10)

    assert ker3.shape[0] == shape3[0], 'Optimal rank of mps is incorrect'
    assert ker2.shape[0] == shape2[0], 'Optimal rank of mps is incorrect'

    tensor_trunc = jnp.tensordot(u, jnp.tensordot(ker1, jnp.tensordot(ker2, ker3, axes=1), axes=1), axes=1)
    half_way_tensor_trunc = jnp.tensordot(ker2, ker3, axes=1)
    half_way_rho_trunc = jnp.tensordot(half_way_tensor_trunc.conj(), half_way_tensor_trunc, axes=[[0], [0]])
    err1 = jnp.linalg.norm(tensor_trunc - tensor_exact)
    err2 = jnp.linalg.norm(half_way_rho_trunc - half_way_rho_exact)
    assert err1 < jnp.sqrt(2) * 1e-10, 'Truncated tensor is not equal to the exact one'
    assert err2 < jnp.sqrt(2) * 1e-10, 'Half-way truncated density matrix is not equal to the exact one'
