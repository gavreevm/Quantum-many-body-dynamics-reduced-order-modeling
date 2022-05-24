"""
Module for decoding reduced order environment state
###################################################################################
Allows:
    - Evironment state decoding
    - Computing composite (System + Environment) energy from reduced order environment
      state
"""

from functools import reduce
import jax.numpy as jnp
from jax import jit

from simulators.exact_simulator_utils import sigma
from simulators.mps import mpo


def statevector_from_ro_state(ro_state, isometries, system_index, number_of_spins):
    """ Returns the decoded system-environment state from reduced state
        Args:
            ro_state: resulting embedding vector
            isometries: isometric matrices from truncation procedure
        Returns: system+environment statevector """

    @jit
    def contract_bot(state, bot_iso):
        state = state.reshape(state.shape[:-1] + (-1, bot_iso.shape[1]))
        state = jnp.tensordot(state, bot_iso, axes=((-1), (1)))
        return state

    @jit
    def contract_top(state, top_iso):
        state = state.reshape((top_iso.shape[1], -1) + state.shape[1:])
        state = jnp.tensordot(top_iso, state, axes=((1), (0)))
        return state

    top, bot = isometries
    if system_index == 0:
        return reduce(contract_bot, bot[::-1], ro_state).reshape(-1)
    elif system_index == number_of_spins - 1:
        return reduce(contract_top, top[::-1], ro_state).reshape(-1)
    else:
        proc_tensor = reduce(contract_top, top[::-1], ro_state)
        return reduce(contract_bot, bot[::-1], proc_tensor).reshape(-1)


def params2hamiltonian_mpo(couplings: jnp.ndarray,
                           fields: jnp.ndarray) -> mpo:
    """ Transform local hamiltonian to MPO form
    Args: couplings (jnp.ndarray) coupling parameters (site, {x, y, z})
          fields (jnp.ndarray) local field parameters (site, {x, y, z})
    Returns:
          Hamiltonian in mpo form (mpo)
    """

    local_parts = jnp.tensordot(fields, sigma, axes=((1), (0)))
    left_vec = jnp.array([local_parts[0]] + list((couplings[0] * sigma.T).T) + [jnp.eye(2)])
    right_vec = jnp.array([jnp.eye(2)] + list(sigma) + [local_parts[-1]])
    zeros = 4 * [jnp.zeros((2, 2), dtype=jnp.complex64)]
    const_rows = jnp.array([[jnp.eye(2)] + zeros, [sigma[0]] + zeros, [sigma[1]] + zeros, [sigma[2]] + zeros])
    mid_blocks = []
    for i in range(1, len(fields) - 1):
        site_row = jnp.array([local_parts[i]] + list((couplings[i] * sigma.T).T) + [jnp.eye(2)])
        block = jnp.concatenate([const_rows, site_row[jnp.newaxis, :]], axis=0)
        mid_blocks.append(block)
    return [left_vec] + mid_blocks + [right_vec]



def renorm_top(top_mpo, top_isometries):
    """Top environment hamiltonian renormalization"""
    @jit
    def mpo_contract_top(stack, oper):
        """ Contract MPO blocks """
        stack_d = stack.shape[-1]
        stack = jnp.tensordot(stack, oper.T, axes=((0), (-1)))
        stack = stack.transpose((4, 0, 3, 1, 2))
        stack = stack.reshape((stack.shape[0],) + 2 * (stack_d * 2,))
        return stack

    def net_contract(state, isometry, mpo):
        process_tensor, i = state # unpacking
        iso_dim, proc_shape = isometry.shape[0], process_tensor.shape
        mpo_num = int(jnp.log2(iso_dim / proc_shape[-1])) # number of additional spins
        process_tensor = reduce(mpo_contract_top, mpo[i:i + mpo_num], process_tensor) # contract mpo
        process_tensor = jnp.tensordot(process_tensor, isometry, axes=((2), (0))) # right isometry
        process_tensor = process_tensor.transpose(0, 2, 1)
        process_tensor = jnp.tensordot(process_tensor, isometry.conj(), axes=((2), (0))) # left isometry
        process_tensor = process_tensor.transpose(0, 2, 1)
        i += mpo_num
        return (process_tensor, i)

    return reduce(lambda state, iso: net_contract(state, iso, top_mpo[1:]),
                            top_isometries, (top_mpo[0], 0))[0]



def renorm_bot(bot_mpo, bot_isometries):
    """Bottom environment hamiltonian renormalization"""
    @jit
    def mpo_contract_bot(stack, oper):
        """ Contract MPO blocks """
        stack_d = stack.shape[-1]
        stack = jnp.tensordot(stack, oper.T, axes=((0), (-2)))
        stack = stack.transpose((4, 3, 0, 2, 1))
        stack = stack.reshape((stack.shape[0],) + 2 * (stack_d * 2,))
        return stack

    def net_contract_bot(state, isometry, mpo):
        process_tensor, i = state # unpacking
        iso_dim, proc_shape = isometry.shape[0], process_tensor.shape
        mpo_num = int(jnp.log2(iso_dim / proc_shape[-1])) # number of additional spins
        process_tensor = reduce(mpo_contract_bot, mpo[i:i + mpo_num], process_tensor) # contract mpo
        process_tensor = jnp.tensordot(process_tensor, isometry, axes=((2), (0))) # right isometry
        process_tensor = process_tensor.transpose(0, 2, 1)
        process_tensor = jnp.tensordot(process_tensor, isometry.conj(), axes=((2), (0))) # left isometry
        process_tensor = process_tensor.transpose(0, 2, 1)
        i += mpo_num
        return (process_tensor, i)

    return reduce(lambda state, iso: net_contract_bot(state, iso, bot_mpo[:-1][::-1]),
                            bot_isometries, (bot_mpo[-1], 0))[0]


def renorm_hamiltonian(ham_mpo, isometries, system_qubit_number):
    """Hamiltinian renormalization for reduced order model energy computation"""
    top_isometries, bot_isometries = isometries
    if system_qubit_number == 0:
        ren_ham = renorm_bot(ham_mpo[1:], bot_isometries)
        ren_bot = jnp.tensordot(ren_ham, ham_mpo[0], axes=((0), (0))).transpose(3, 1, 2, 0)
        return jnp.expand_dims(ren_bot, (0, 3))

    if system_qubit_number == len(ham_mpo) - 1:
        ren_ham = renorm_top(ham_mpo[:-1], top_isometries)
        ren_top = jnp.tensordot(ren_ham, ham_mpo[0], axes=((0), (0))).transpose(1, 3, 0, 2)
        return jnp.expand_dims(ren_top, (2, 5))

    bot_ren_ham = renorm_bot(ham_mpo[(system_qubit_number + 1):], bot_isometries)
    top_ren_ham = renorm_top(ham_mpo[:system_qubit_number], top_isometries)
    ren_ham = jnp.tensordot(top_ren_ham, ham_mpo[system_qubit_number], axes=((0), (0)))
    ren_ham = jnp.tensordot(ren_ham, bot_ren_ham, axes=((2), (0)))
    return ren_ham.transpose(1, 3, 5, 0, 2, 4)


def environment_energy(ren_ham, final_state):
    """Calculate energy of the environment """
    energy = jnp.tensordot(ren_ham, final_state, axes=((0, 1, 2), (0, 1, 2)))
    energy = jnp.tensordot(energy, final_state.conj(), axes=((0, 1, 2), (0, 1, 2)))
    return jnp.real(energy)


def _bond_ham(couplings, fields):
    bond_pars = []
    n = len(couplings)
    zeros = jnp.zeros(3)
    for i in range(n):
        bond_couplings = jnp.array(i * [jnp.zeros(3)] +[couplings[i]] + (n - i - 1) * [jnp.zeros(3)])
        bond_fields = jnp.array(i * [zeros] + [fields[i]] + [fields[i+1]] + (n - i - 1) * [zeros])
        bond_pars.append((bond_couplings, bond_fields))
    return bond_pars


def local_hamiltonian_renormalization(isometries, couplings, fields, system_qubit_number):
    """ Renormalized operators to calculate local energy density """
    def renorm(bond_pars):
        coups, flds = bond_pars
        bond_mpo = params2hamiltonian_mpo(coups, flds)
        return renorm_hamiltonian(bond_mpo, isometries, system_qubit_number)

    bond_pars_list = _bond_ham(couplings, fields)
    return list(map(renorm, bond_pars_list))
