import os
#os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

from simulators.logger import load_params, load_data
from experiments_utils import rho2bloch
from simulators.exact_simulator_utils import M

data = {}
for i, (root, dirs, files) in enumerate(os.walk("./experiment2_data")):
    if i > 0:
        params = load_params(os.path.join(root, 'params.txt'))
        initial_state = jnp.array(params['system_state'])
        initial_state = jnp.tensordot(initial_state, initial_state.conj(), axes=0)
        initial_state = initial_state[jnp.newaxis]
        first_row = 2 * jnp.log(2) * jax.nn.one_hot(jnp.array(params['source_qubit']), params['n'])[jnp.newaxis]
        controlled_mutual_information = jnp.concatenate([first_row, load_data(os.path.join(root, 'controlled_mutual_information.pickle'))], axis=0)
        controlled_quantum_channels = load_data(os.path.join(root, 'controlled_quantum_channels.pickle'))
        input_states = M.T.reshape((-1, 2, 2))
        output_states = jnp.tensordot(input_states, controlled_quantum_channels[-1, params['system_qubit']], axes=[[1, 2], [-2, -1]])
        input_bloch_vecs = rho2bloch(input_states)
        output_bloch_vecs = rho2bloch(output_states)

        # plotting mutual information dynamics under optimal control
        plt.figure()
        plt.imshow(controlled_mutual_information.T, cmap='Spectral')
        plt.savefig('./plots_experiment2/controlled_mutual_information_{}_{}_{}.pdf'.format(params['n'], params['system_qubit'], params['source_qubit']))

        # plotting input/output states
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect(aspect = (1, 1, 1))
        ax._axis3don = False

        u = jnp.linspace(0, 2 * jnp.pi, 100)
        v = jnp.linspace(0, jnp.pi, 100)

        x = 1 * jnp.outer(jnp.cos(u), jnp.sin(v))
        y = 1 * jnp.outer(jnp.sin(u), jnp.sin(v))
        z = 1 * jnp.outer(jnp.ones(jnp.size(u)), jnp.cos(v))

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.1)

        ax.scatter(input_bloch_vecs[:, 0], input_bloch_vecs[:, 1], input_bloch_vecs[:, 2], color='blue', marker='o', label=r'${\rm Input \ qubit \ state}$')
        ax.scatter(output_bloch_vecs[:, 0], output_bloch_vecs[:, 1], output_bloch_vecs[:, 2], color='red', marker='^', label=r'${\rm Output \ qubit \ state}$')
        ax.plot([input_bloch_vecs[0, 0], output_bloch_vecs[0, 0]],
                [input_bloch_vecs[0, 1], output_bloch_vecs[0, 1]],
                [input_bloch_vecs[0, 2], output_bloch_vecs[0, 2]], ':r')
        ax.plot([input_bloch_vecs[1, 0], output_bloch_vecs[1, 0]],
                [input_bloch_vecs[1, 1], output_bloch_vecs[1, 1]],
                [input_bloch_vecs[1, 2], output_bloch_vecs[1, 2]], ':r')
        ax.plot([input_bloch_vecs[2, 0], output_bloch_vecs[2, 0]],
                [input_bloch_vecs[2, 1], output_bloch_vecs[2, 1]],
                [input_bloch_vecs[2, 2], output_bloch_vecs[2, 2]], ':r')
        ax.plot([input_bloch_vecs[3, 0], output_bloch_vecs[3, 0]],
                [input_bloch_vecs[3, 1], output_bloch_vecs[3, 1]],
                [input_bloch_vecs[3, 2], output_bloch_vecs[3, 2]], ':r')
        
        ax.legend(frameon=False, loc=[0.3, 0.8])

        plt.savefig('./plots_experiment2/input_output_states_{}_{}_{}.pdf'.format(params['n'], params['system_qubit'], params['source_qubit']))
