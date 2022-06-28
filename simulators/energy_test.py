import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
from functools import reduce
from tqdm import tqdm


identity = jnp.array([[1., 0.], [0., 1.]], jnp.complex64)
pauli = [jnp.array([[0., 1.], [1., 0.]], jnp.complex64),
         jnp.array([[0., -1j], [1j, 0.]], jnp.complex64),
         jnp.array([[1., 0.], [0., -1.]], jnp.complex64)]


def exact_hamiltonian(couplings, fields):
    """
    ==============================================================
                        | Test function |
    Do not use for large dimentions to prevent RAM overconsumption
    ==============================================================
    Compute exact hamiltonian

    Input: coupling parameters (J), local fields (h)
    Output: exact hamiltonian
    """

    def contract(coeffs, oper_list):
        return jnp.tensordot(coeffs,
               jnp.array([[reduce(lambda x, y: jnp.kron(x, y), element)
                                  for element in oper_arr]
                                  for oper_arr in oper_list]),
                                  axes=((0, 1), (1, 0)))

    sites_number = fields.shape[0]
    coup_list = jnp.array([[(i) * [identity] + 2 * [operator] + \
                            (sites_number - i - 2) * [identity]
                            for i in range(sites_number - 1)]
                            for operator in pauli])
    fields_list = jnp.array([[(i) * [identity] + [operator] + \
                              (sites_number - i - 1) * [identity]
                              for i in range(sites_number)]
                              for operator in pauli])
    return contract(couplings, coup_list) + \
                contract(fields, fields_list)


def exact_dynamics(hamiltonian, initial_state, tau, time_steps,
                   use_control=False, control_seq=None):
    """ Calculate exact dynamics of the whole system
        ==============================================================
                        | Test function |
        Do not use for large dimentions to prevent RAM overconsumption
        ==============================================================
    """
    unitary = expm(-1j * tau * hamiltonian)
    conseq_states = [initial_state]
    dim = int(initial_state.shape[0] / 2)
    if use_control == False:
        for i in tqdm(range(time_steps), ncols=100):
            initial_state = jnp.matmul(unitary, initial_state)
            conseq_states.append(initial_state)
    else:
        for control in tqdm(control_seq):
            initial_state = jnp.matmul(unitary, initial_state)
            c = jnp.kron(control, jnp.eye(dim))
            initial_state = jnp.matmul(c, initial_state)
            conseq_states.append(initial_state)
    return jnp.array(conseq_states)


@jit
def measure(density_matrix):
    return [(density_matrix @ pauli[0]).trace(),
            (density_matrix @ pauli[1]).trace(),
            (density_matrix @ pauli[2]).trace()]


def trace_environment(statevector, spins_number):
    """ Tracing the exct dynamics
        ==============================================================
                        | Test function |
        Do not use for large dimentions to prevent RAM overconsumption
        ==============================================================
    """
    meas_res = []
    spin_range = jnp.arange(spins_number)
    traces = [tuple(jnp.delete(spin_range, index)) for index in spin_range]
    resh_vector = statevector.reshape(spins_number * [2])
    for trace in traces:
        traced_density = jnp.tensordot(resh_vector, resh_vector.conj(),
                                                  axes=(trace, trace))
        meas_res.append(measure(traced_density))
    return meas_res

