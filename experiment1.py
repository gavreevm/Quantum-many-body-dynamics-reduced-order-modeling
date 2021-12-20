import os
from functools import reduce
import uuid
import jax.numpy as jnp
import matplotlib.pyplot as plt

from simulators.logger import save_params, save_data
from simulators.models_utils import params2gates_layer
from simulators.exact_simulator import ExactSimulator
from simulators.reduced_order_simulator import ReducedOrderSimulator
from simulators.exact_simulator_utils import sigma

# This file performs simulation of the system using both the exact
# simulator and the reduced-order simulator with zero control signal.

# ------------------ Description of model parameters ------------------------ #
# N --------------------- number of discrete time steps
# n --------------------- number of spins
# tau ------------------- time step duration
# hi -------------------- i-th component of the external magnetic field (i \in {x, y, z})
# Ji -------------------- i-th component of the coupling (i \in {x, y, z})
# system_qubit ---------- the number of the system spin (ranges from 0, to n-1)
# system_state ---------- psi vector of the system spin
# env_single_spin_state - psi vector of each spin in the environment
# eps ------------------- accuracy of truncation
# full_truncation ------- flag showing whether to use full truncation or not

experiment_id = str(uuid.uuid4())
dir_path = 'experiment1_data/' + experiment_id
os.mkdir(dir_path)

params = {
    'N' : 50,
    'n' : 11,
    'tau' : 0.15,
    'hx' : 0.2,
    'hy' : 0.2,
    'hz' : 0.2,
    'Jx' : 0.9,
    'Jy' : 1,
    'Jz' : 1.1,
    'system_qubit' : 0,
    'system_state' : [1, 0],
    'env_single_spin_state' : [0, 1],
    'eps' : 1e-2,
    'full_truncation' : True,
    'truncate_when' : 512,
}
save_params(params, dir_path + '/params.txt')

# gates
gates_layer = params2gates_layer(params)
save_data(gates_layer, dir_path + '/gates_layer.pickle')

# initial env. state for the reduced-order simulator
ro_env_state = (params['n'] - 1) * [jnp.array(params['env_single_spin_state'], dtype=jnp.complex64)]

# initial env. state for the exact simulator
ex_env_state = reduce(lambda x, y: jnp.kron(x, y), ro_env_state)

# system state
system_state = jnp.array(params['system_state'], dtype=jnp.complex64)

# zero control
control_gates = jnp.tile(jnp.eye(2, dtype=jnp.complex64)[jnp.newaxis], (params['N'], 1, 1))

# EXACT DYNAMICS SIMULATION
ex_sim = ExactSimulator()
ex_sim_state = ex_sim.initialize(params['n'],
                                 params['system_qubit'],
                                 params['system_qubit'],
                                 params['N'])
quantum_channels = ex_sim.compute_quantum_channels(ex_sim_state,
                                                   ex_env_state,
                                                   gates_layer,
                                                   control_gates)
exact_density_matrices = jnp.einsum('ijklmn,m,n->ijkl', quantum_channels, system_state, system_state.conj())
save_data(quantum_channels, dir_path + '/quantum_channels.pickle')
save_data(exact_density_matrices, dir_path + '/exact_density_matrices.pickle')

# REDUCED_ORDER MODEL BASED DYNAMICS SIMULATION
ro_sim = ReducedOrderSimulator()
ro_model = ro_sim.build_reduced_order_model(params['system_qubit'],
                                            params['system_qubit'],
                                            params['N'],
                                            ro_env_state,
                                            gates_layer,
                                            params['full_truncation'],
                                            params['truncate_when'],
                                            params['eps'])
save_data(ro_model, dir_path + '/ro_model.pickle')
ro_model_based_density_matrices = ro_sim.compute_dynamics(ro_model, control_gates, system_state)
save_data(ro_model_based_density_matrices, dir_path + '/ro_based_density_matrices.pickle')

#SIMPLE PLOTTING

ro_bloch_vectors = jnp.tensordot(ro_model_based_density_matrices, sigma, axes=[[1, 2], [2, 1]])
exact_bloch_vectors = jnp.tensordot(exact_density_matrices[:, params['system_qubit']], sigma, axes=[[1, 2], [2, 1]])

plt.figure()
plt.plot(ro_bloch_vectors[:, 0], 'r')
plt.plot(ro_bloch_vectors[:, 1], 'b')
plt.plot(ro_bloch_vectors[:, 2], 'k')
plt.plot(exact_bloch_vectors[:, 0], '*r')
plt.plot(exact_bloch_vectors[:, 1], 'ob')
plt.plot(exact_bloch_vectors[:, 2], 'xk')
plt.legend(['exact x', 'exact y', 'exact z', 'ro x', 'ro y', 'ro z'])
plt.ylabel('Amplitude')
plt.xlabel('N')
plt.savefig(dir_path + '/dynamics.pdf')

plt.figure()
plt.plot([ker.shape[0] + ker.shape[2] for ker in reversed(ro_model)])
plt.ylabel('Environment dimension')
plt.xlabel('N')
plt.savefig(dir_path + '/env_dimension.pdf')
