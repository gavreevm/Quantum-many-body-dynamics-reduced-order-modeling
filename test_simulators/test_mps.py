from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import pytest
import copy

import jax.numpy as jnp
from jax import random
from simulators.mps import set_to_forward_canonical, set_to_backward_canonical, dot_prod, truncate_forward_canonical, truncate_very_last_edge_backward_canonical

from functools import reduce


mps_ker_num = 4


key = random.PRNGKey(42)
subkey0, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)


bond_dims = 2 + random.categorical(subkey1, jnp.ones((5,)), shape=(2, mps_ker_num-1))
bond_dims = jnp.concatenate([5*jnp.ones((2, 1), dtype=jnp.int32), bond_dims, 2*jnp.ones((2, 1), dtype=jnp.int32)], axis=-1)
dims = random.categorical(subkey2, jnp.ones((3,)), shape=(mps_ker_num,)) + 2
shapes1 = [(left_bond, dim, right_bond) for left_bond, dim, right_bond in zip(bond_dims[0, :-1], dims, bond_dims[0, 1:])]
shapes2 = [(left_bond, dim, right_bond) for left_bond, dim, right_bond in zip(bond_dims[1, :-1], dims, bond_dims[1, 1:])]
inp_mps1 = [(lambda x: x[..., 0] + 1j * x[..., 1])(random.normal(key, (*shape, 2))) for shape, key in zip(shapes1, random.split(subkey3, len(shapes1)))]
inp_mps2 = [(lambda x: x[..., 0] + 1j * x[..., 1])(random.normal(key, (*shape, 2))) for shape, key in zip(shapes2, random.split(subkey4, len(shapes2)))]


def generate_low_rank_ker(key, shape):
    subkey1, subkey2 = random.split(key)
    ker = random.normal(subkey1, (*shape, 2))
    ker = ker[..., 0] + 1j * ker[..., 1]
    irrelevant_ker = random.normal(subkey2, (shape[0], shape[-1], 2))
    irrelevant_ker = irrelevant_ker[..., 0] + 1j * irrelevant_ker[..., 1]
    ker = jnp.tensordot(ker, irrelevant_ker, axes=0)
    ker = ker.transpose((0, 3, 1, 2, 4))
    ker = ker.reshape((shape[0] ** 2, shape[1], shape[-1] ** 2))
    return ker
shapes0 = [(1, dims[0], 2)] + [(2, dim, 2) for dim in dims[1:-1]] + [(2, dims[-1], 1)]
low_rank_mps = [generate_low_rank_ker(key, shape) for shape, key in zip(shapes0, random.split(subkey0, mps_ker_num))]


def log_cos_dist(inp_mps1, inp_mps2):
    term1 = dot_prod(inp_mps1, inp_mps2).real
    term2 = dot_prod(inp_mps1, inp_mps1).real
    term3 = dot_prod(inp_mps2, inp_mps2).real
    return 0.5 * (term2 + term3) - term1


def get_tensor(inp_mps):
    def iter(state, ker):
        state = jnp.tensordot(ker, state, axes=1)
        return state
    return reduce(iter, reversed(inp_mps), jnp.eye(inp_mps[-1].shape[-1]))


@pytest.mark.parametrize("inp_mps1,inp_mps2", [(inp_mps1, inp_mps2)])
def test_dot_prod(inp_mps1, inp_mps2):
    """One calculate dot prod via naive method and via mps dot product and then
    compare"""

    def iter(state, ker):
        state = jnp.tensordot(ker, state, axes=1)
        return state
    tensor1 = get_tensor(inp_mps1).reshape((-1,))
    tensor2 = get_tensor(inp_mps2).reshape((-1,))
    naive_dot_prod = jnp.tensordot(tensor1, tensor2.conj(), axes=1)
    mps_dot_prod = jnp.exp(dot_prod(inp_mps1, inp_mps2))
    err = jnp.abs(naive_dot_prod - mps_dot_prod)
    assert err < 1e-10, 'Mps based dot product does not match the naive dot product'


@pytest.mark.parametrize("inp_mps", [inp_mps1])
def test_set_to_forward_canonical(inp_mps):
    """Check variou properties of the obtained canonical form"""

    inp_mps_copy = copy.deepcopy(inp_mps)
    log_norm_via_dot_prod = dot_prod(inp_mps_copy, inp_mps_copy) / 2
    log_norm_via_forward_canonical = set_to_forward_canonical(inp_mps_copy)
    err1 = jnp.abs(log_norm_via_dot_prod - log_norm_via_forward_canonical)
    for i, v in enumerate(inp_mps_copy[:-1]):
        vvh = jnp.tensordot(v, v.conj(), axes=[[0, 1], [0, 1]])
        err = jnp.linalg.norm(vvh - jnp.eye(vvh.shape[0]))
        assert err < 1e-10, 'Kernel number {} is not isometric'.format(i)

    err2 = log_cos_dist(inp_mps_copy, inp_mps)

    assert err1 < 1e-10, 'Norm of the canonical form does not mach the norm of the initial mps'
    assert err2 < 1e-10, 'Angle between the cnanonical form and the initial mps is too big'


@pytest.mark.parametrize("inp_mps", [inp_mps1])
def test_set_to_backward_canonical(inp_mps):
    """Check various properties of the obtained canonical form"""

    inp_mps_copy = copy.deepcopy(inp_mps)
    log_norm_via_dot_prod = dot_prod(inp_mps_copy, inp_mps_copy) / 2
    log_norm_via_backward_canonical = set_to_backward_canonical(inp_mps_copy)
    err1 = jnp.abs(log_norm_via_dot_prod - log_norm_via_backward_canonical)
    for i, v in enumerate(inp_mps_copy[1:]):
        vhv = jnp.tensordot(v, v.conj(), axes=[[1, 2], [1, 2]])
        err = jnp.linalg.norm(vhv - jnp.eye(vhv.shape[0]))
        assert err < 1e-10, 'Kernel number {} is not isometric'.format(i)
    
    err2 = log_cos_dist(inp_mps_copy, inp_mps)

    assert err1 < 1e-10, 'Norm of the canonical form does not mach the norm of the initial mps'
    assert err2 < 1e-10, 'Angle between the cnanonical form and the initial mps is too big'


@pytest.mark.parametrize("inp_mps,low_rank_shapes", [(low_rank_mps, shapes0)])
def test_truncate_forward_canonical(inp_mps, low_rank_shapes):
    inp_mps_copy = copy.deepcopy(inp_mps)
    log_norm = set_to_forward_canonical(inp_mps_copy)
    truncate_forward_canonical(inp_mps_copy, 1e-10)
    for i, (ker, shape) in enumerate(zip(inp_mps_copy, low_rank_shapes)):
        assert ker.shape == shape, 'Kernel number {} has incorrect shape'.format(i)

    err = log_cos_dist(inp_mps_copy, inp_mps)

    assert err < 1e-10, 'Angle between the truncated mps and the initial mps is too big'


@pytest.mark.parametrize("inp_mps", [(low_rank_mps[1:])])
def test_truncate_very_last_edge_backward_canonical(inp_mps):
    inp_mps_copy = copy.deepcopy(inp_mps)
    lognorm = set_to_backward_canonical(inp_mps_copy)
    truncate_very_last_edge_backward_canonical(inp_mps_copy, 1e-10)
    tensor1 = get_tensor(inp_mps)
    tensor2 = jnp.exp(lognorm) * get_tensor(inp_mps_copy)
    tensor1 = jnp.tensordot(tensor1.conj(), tensor1, axes=[[0], [0]]).reshape((-1,))
    tensor2 = jnp.tensordot(tensor2.conj(), tensor2, axes=[[0], [0]]).reshape((-1,))
    err = jnp.linalg.norm(tensor1 - tensor2) / jnp.linalg.norm(tensor1)
    assert err < 1e-10, 'The distance between truncated density matrix and the exact one is too big'
