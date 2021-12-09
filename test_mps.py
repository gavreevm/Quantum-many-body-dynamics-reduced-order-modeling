from jax.config import config
config.update("jax_enable_x64", True)

import pytest
import copy

import jax.numpy as jnp
from jax import random
from mps import set_to_forward_canonical, set_to_backward_canonical, dot_prod

from functools import reduce

mps_ker_num = 4

key = random.PRNGKey(42)
key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

bond_dims = 2 + random.categorical(subkey1, jnp.ones((5,)), shape=(2, mps_ker_num-1))
bond_dims = jnp.concatenate([5*jnp.ones((2, 1), dtype=jnp.int32), bond_dims, 2*jnp.ones((2, 1), dtype=jnp.int32)], axis=-1)
dims = random.categorical(subkey2, jnp.ones((3,)), shape=(mps_ker_num,)) + 2
shapes1 = [(left_bond, dim, right_bond) for left_bond, dim, right_bond in zip(bond_dims[0, :-1], dims, bond_dims[0, 1:])]
shapes2 = [(left_bond, dim, right_bond) for left_bond, dim, right_bond in zip(bond_dims[1, :-1], dims, bond_dims[1, 1:])]
inp_mps1 = [(lambda x: x[..., 0] + 1j * x[..., 1])(random.normal(key, (*shape, 2)))for shape, key in zip(shapes1, random.split(subkey3, len(shapes1)))]
inp_mps2 = [(lambda x: x[..., 0] + 1j * x[..., 1])(random.normal(key, (*shape, 2)))for shape, key in zip(shapes2, random.split(subkey4, len(shapes2)))]


@pytest.mark.parametrize("inp_mps1,inp_mps2", [(inp_mps1, inp_mps2)])
def test_dot_prod(inp_mps1, inp_mps2):
    """One calculate dot prod via naive method and via mps dot product and then
    compare"""

    def iter(state, ker):
        state = jnp.tensordot(ker, state, axes=1)
        return state
    tensor1 = reduce(iter, reversed(inp_mps1), jnp.eye(inp_mps1[-1].shape[-1])).reshape((-1,))
    tensor2 = reduce(iter, reversed(inp_mps2), jnp.eye(inp_mps2[-1].shape[-1])).reshape((-1,))
    naive_dot_prod = jnp.tensordot(tensor1, tensor2.conj(), axes=1)
    mps_dot_prod = jnp.exp(dot_prod(inp_mps1, inp_mps2))
    err = jnp.abs(naive_dot_prod - mps_dot_prod)
    assert err < 1e-10, 'Mps based dot product does not match the naive dot product'


@pytest.mark.parametrize("inp_mps", [inp_mps1])
def test_set_to_forward_canonical(inp_mps):
    """Check variou properties of the obtained canonical form"""

    inp_mps_copy = copy.deepcopy(inp_mps)
    log_norm_via_dot_prod = dot_prod(inp_mps_copy, inp_mps_copy) / 2
    log_norm_via_forward_canonical, r = set_to_forward_canonical(inp_mps_copy)
    err1 = jnp.abs(log_norm_via_dot_prod - log_norm_via_forward_canonical)
    for i, v in enumerate(inp_mps_copy):
        vvh = jnp.tensordot(v, v.conj(), axes=[[0, 1], [0, 1]])
        err = jnp.linalg.norm(vvh - jnp.eye(vvh.shape[0]))
        assert err < 1e-10, 'Kernel number {} is not isometric'.format(i)
    inp_mps_copy[-1] = jnp.tensordot(inp_mps_copy[-1], r, axes=1)
    err2 = jnp.exp(2 * log_norm_via_dot_prod.real) + jnp.exp((dot_prod(inp_mps, inp_mps)).real) - 2 * jnp.exp(dot_prod(inp_mps_copy, inp_mps) + log_norm_via_forward_canonical).real

    assert err1 < 1e-10, 'Norm of the canonical form does not mach the norm of the initial mps'
    assert err2 < 1e-10, 'Cnanonical form and the initial mps represent different tensors'


@pytest.mark.parametrize("inp_mps", [inp_mps1])
def test_set_to_backward_canonical(inp_mps):
    """Check variou properties of the obtained canonical form"""

    inp_mps_copy = copy.deepcopy(inp_mps)
    log_norm_via_dot_prod = dot_prod(inp_mps_copy, inp_mps_copy) / 2
    log_norm_via_backward_canonical, r = set_to_backward_canonical(inp_mps_copy)
    err1 = jnp.abs(log_norm_via_dot_prod - log_norm_via_backward_canonical)
    for i, v in enumerate(inp_mps_copy):
        vhv = jnp.tensordot(v, v.conj(), axes=[[1, 2], [1, 2]])
        err = jnp.linalg.norm(vhv - jnp.eye(vhv.shape[0]))
        assert err < 1e-10, 'Kernel number {} is not isometric'.format(i)
    inp_mps_copy[0] = jnp.tensordot(r, inp_mps_copy[0], axes=1)
    err2 = jnp.exp(2 * log_norm_via_dot_prod.real) + jnp.exp((dot_prod(inp_mps, inp_mps)).real) - 2 * (jnp.exp(dot_prod(inp_mps_copy, inp_mps) + log_norm_via_backward_canonical)).real

    assert err1 < 1e-10, 'Norm of the canonical form does not mach the norm of the initial mps'
    assert err2 < 1e-10, 'Cnanonical form and the initial mps represent different tensors'
