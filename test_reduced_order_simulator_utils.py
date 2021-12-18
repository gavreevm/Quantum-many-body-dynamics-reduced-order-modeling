from jax.config import config
config.update("jax_enable_x64", True)

import copy
from functools import reduce

from jax import numpy as jnp
from jax import random
from mps import mpo_mps_product, dot_prod
from reduced_order_simulator_utils import _layer2lattice
from exact_simulator import ExactSimulator


def test_layer2lattice():
    """This test compares evolution operator calculated from a lattice with
    evolution operator calculated directly"""

    key = random.PRNGKey(42)

    number_of_qubits = 7
    discrete_time = 7

    gates_layer = random.normal(key, (number_of_qubits-1, 4, 4, 2))
    gates_layer = gates_layer[..., 0] + 1j * gates_layer[..., 1]
    gates_layer, _ = jnp.linalg.qr(gates_layer)

    lattice = _layer2lattice(gates_layer, discrete_time)

    for i, ker in enumerate(lattice[0]):
        conv1 = jnp.einsum('ijk,ljk->il', ker, ker.conj())
        conv1 /= jnp.trace(conv1)
        conv1 *= conv1.shape[0]
        err1 = jnp.linalg.norm(conv1 - jnp.eye(conv1.shape[0]))
        conv2 = jnp.einsum('jki,jkl->il', ker, ker.conj())
        conv2 /= jnp.trace(conv2)
        conv2 *= conv2.shape[0]
        err2 = jnp.linalg.norm(conv2 - jnp.eye(conv2.shape[0]))
        assert err1 < 1e-10, "Ker #{} of top mps is not right isometric".format(i)
        assert err2 < 1e-10, "Ker #{} of top mps is not left isometric".format(i)

    for i, mpo in enumerate(lattice[1:-1]):
        for j, ker in enumerate(mpo):
            conv1 = jnp.einsum('ijkl,njkl->in', ker, ker.conj())
            conv1 /= jnp.trace(conv1)
            conv1 *= conv1.shape[0]
            err1 = jnp.linalg.norm(conv1 - jnp.eye(conv1.shape[0]))
            conv2 = jnp.einsum('jkil,jknl->in', ker, ker.conj())
            conv2 /= jnp.trace(conv2)
            conv2 *= conv2.shape[0]
            err2 = jnp.linalg.norm(conv2 - jnp.eye(conv2.shape[0]))
            assert err1 < 1e-10, "Ker #{} of mpo #{} is not right isometric".format(j, i)
            assert err2 < 1e-10, "Ker #{} of mpo #{} is not left isometric".format(j, i)

    for i, ker in enumerate(lattice[-1]):
        conv1 = jnp.einsum('ijk,ljk->il', ker, ker.conj())
        conv1 /= jnp.trace(conv1)
        conv1 *= conv1.shape[0]
        err1 = jnp.linalg.norm(conv1 - jnp.eye(conv1.shape[0]))
        conv2 = jnp.einsum('jki,jkl->il', ker, ker.conj())
        conv2 /= jnp.trace(conv2)
        conv2 *= conv2.shape[0]
        err2 = jnp.linalg.norm(conv2 - jnp.eye(conv2.shape[0]))
        assert err1 < 1e-10, "Ker #{} of bottom mps is not right isometric".format(i)
        assert err2 < 1e-10, "Ker #{} of bottom mps is not left isometric".format(i)

    mps = copy.deepcopy(lattice[0])

    for mpo in lattice[1:-1]:
        mpo_mps_product(mpo, mps)

    mps_mps = lambda x, y: jnp.einsum('ijk,ljm->ilkm', x, y).reshape((2 ** number_of_qubits, 2 ** number_of_qubits))
    partly_contructed = [mps_mps(*args) for args in zip(mps, lattice[-1])]

    for i, (u1, u2) in enumerate(zip(partly_contructed[:-1], partly_contructed[1:])):
        err = jnp.linalg.norm(u1 - u2)
        assert err < 1e-10, "u_{} and u_{} are not equal".format(i, i+1)

    U1 = reduce(jnp.kron, gates_layer[2::2], gates_layer[0])
    U1 = jnp.kron(U1, jnp.eye(2))
    U2 = reduce(jnp.kron, gates_layer[3::2], gates_layer[1])
    U2 = jnp.kron(jnp.eye(2), U2)
    err = jnp.linalg.norm(partly_contructed[0] - U2 @ U1)
    assert err < 1e-10 , "Resulting circuit leads to incorrect evolution operator"
