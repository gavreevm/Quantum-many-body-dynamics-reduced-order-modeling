import os
#os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin")
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

from simulators.logger import load_params, load_data
from experiments_utils import rho2bloch

def latex(x):
    return list(map(lambda y: r'${}$'.format(y), x))

data = {}
for i, (root, dirs, files) in enumerate(os.walk("./experiment1_data")):
    if i > 0:
        subdata = {}
        params = load_params(os.path.join(root, 'params.txt'))
        tau = params['tau']
        N = params['N']
        n = params['n']
        initial_state = jnp.array(params['system_state'])
        initial_state = jnp.tensordot(initial_state, initial_state.conj(), axes=0)
        initial_state = initial_state[jnp.newaxis]
        system_qubit = params['system_qubit']
        first_row = 2 * jnp.log(2) * jax.nn.one_hot(jnp.array(system_qubit), n)[jnp.newaxis]
        # there is a small mistake in the uploaded mutual information, thus it requires slight rescaling an shift
        subdata['controlled_mutual_information'] = jnp.concatenate([first_row, (load_data(os.path.join(root, 'controlled_mutual_information.pickle')) + 2 * jnp.log(2)) / 2], axis=0)
        subdata['zero_control_mutual_information'] = jnp.concatenate([first_row, (load_data(os.path.join(root, 'zero_control_mutual_information.pickle')) + 2 * jnp.log(2)) / 2], axis=0)
        subdata['exact_dynamics_zero_control'] = rho2bloch(jnp.concatenate([initial_state, load_data(os.path.join(root, 'zero_control_exact_density_matrices.pickle'))[:, system_qubit]], axis=0))
        subdata['ro_based_dynamics_zero_control'] = rho2bloch(jnp.concatenate([initial_state, load_data(os.path.join(root, 'zero_control_ro_based_density_matrices.pickle'))], axis=0))
        subdata['exact_dynamics_controlled'] = rho2bloch(jnp.concatenate([initial_state, load_data(os.path.join(root, 'controlled_exact_density_matrices.pickle'))[:, system_qubit]], axis=0))
        subdata['ro_based_dynamics_controlled'] = rho2bloch(jnp.concatenate([initial_state, load_data(os.path.join(root, 'controlled_ro_based_density_matrices.pickle'))], axis=0))
        subdata['ro_dim'] = list(map(lambda x: 2 * x[0].shape[0] * x[2].shape[0], load_data(os.path.join(root, 'ro_model.pickle'))))[::-1]
        subdata['params'] = params
        data[(n, system_qubit)] = subdata

plots_num = len(data)
plt.figure(figsize=(10, 20))
plt.subplots_adjust(hspace = 0.12, wspace = 0.06)
for i, (key, value) in enumerate(sorted(data.items(), key=lambda kv: kv[0])):
    ax = plt.subplot(plots_num // 2, 4, 2*i+1)
    if i < 2:
        ax.set_title(r'${\rm Control \ OFF}$', fontsize=18)
    plt.xticks([0, 4, 9, 14, 19, 24], latex([1, 5, 10, 15, 20, 25]))
    if (i % 2 == 0):
        plt.yticks([i * (N / 3) for i in range(4)], latex([round(tau * i * (N / 3), 2) for i in range(4)]))
        plt.ylabel(r'${\rm Time}$', fontsize=18)   
    else:
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
    if i // 2 == plots_num // 2 - 1:
       plt.xlabel(r'${\rm Spin \ number}$', fontsize=18)   
    plt.imshow(value['zero_control_mutual_information'], aspect='auto', interpolation=None, cmap='Spectral')
    ax = plt.subplot(plots_num // 2, 4, 2*i+2)
    if i < 2:
        ax.set_title(r'${\rm Control \ ON}$', fontsize=18)
    plt.xticks([0, 4, 9, 14, 19, 24], latex([1, 5, 10, 15, 20, 25]))
    ax = plt.gca()
    if i // 2 == plots_num // 2 - 1:
        plt.xlabel(r'${\rm Spin \ number}$', fontsize=18)
    ax.axes.yaxis.set_visible(False)
    if i % 2 == 1:
        ax = ax.twinx()
        plt.yticks([], [])
        plt.ylabel(r'$n = {}$'.format(key[0]), fontsize=18)
    plt.imshow(value['controlled_mutual_information'], aspect='auto', interpolation=None, cmap='Spectral')
plt.savefig('plots_experiment1/mutual_information_flows.pdf')

plt.figure(figsize=(10, 7.5))
plt.subplots_adjust(hspace = 0.10, wspace = 0.15)
ax = plt.subplot(2, 2, 1)
ax.set_title(r'${1{\rm st \ spin \ tracking}}$', fontsize=18)
plt.ylim(top=1.1, bottom=-1.1)
plt.xticks(list(map(lambda x: x / tau, [0, 1, 2, 3, 4, 5, 6, 7])), latex([0, 1, 2, 3, 4, 5, 6, 7]))
plt.plot(data[(27, 0)]['exact_dynamics_zero_control'][:, 0], 'r')
plt.plot(data[(27, 0)]['exact_dynamics_zero_control'][:, 1], 'b')
plt.plot(data[(27, 0)]['exact_dynamics_zero_control'][:, 2], 'k')
plt.plot(data[(27, 0)]['ro_based_dynamics_zero_control'][:, 0], '+r')
plt.plot(data[(27, 0)]['ro_based_dynamics_zero_control'][:, 1], 'xb')
plt.plot(data[(27, 0)]['ro_based_dynamics_zero_control'][:, 2], 'ok')
plt.legend([r'$\langle\sigma_x\rangle \ {\rm exact}$', r'$ \langle\sigma_y\rangle\ {\rm exact}$', r'$\langle\sigma_z\rangle \ {\rm exact}$',
            r'$\langle\sigma_x\rangle \ {\rm reduced-order}$', r'$\langle\sigma_y\rangle \ {\rm reduced-order}$', r'$\langle\sigma_z\rangle \ {\rm reduced-order}$'], ncol=2, frameon=False)
ax=plt.subplot(2, 2, 2)
ax.set_title(r'${14{\rm th \ spin \ tracking}}$', fontsize=18)
plt.ylim(top=1.1, bottom=-1.1)
plt.xticks(list(map(lambda x: x / tau, [0, 1, 2, 3, 4, 5, 6, 7])), latex([0, 1, 2, 3, 4, 5, 6, 7]))
plt.plot(data[(27, 13)]['exact_dynamics_zero_control'][:, 0], 'r')
plt.plot(data[(27, 13)]['exact_dynamics_zero_control'][:, 1], 'b')
plt.plot(data[(27, 13)]['exact_dynamics_zero_control'][:, 2], 'k')
plt.plot(data[(27, 13)]['ro_based_dynamics_zero_control'][:, 0], '+r')
plt.plot(data[(27, 13)]['ro_based_dynamics_zero_control'][:, 1], 'xb')
plt.plot(data[(27, 13)]['ro_based_dynamics_zero_control'][:, 2], 'ok')
plt.subplot(2, 2, 3)
plt.xticks(list(map(lambda x: x / tau, [0, 1, 2, 3, 4, 5, 6, 7])), latex([0, 1, 2, 3, 4, 5, 6, 7]))
plt.ylim(top=1.1, bottom=-1.1)
plt.plot(data[(27, 0)]['exact_dynamics_controlled'][:, 0], 'r')
plt.plot(data[(27, 0)]['exact_dynamics_controlled'][:, 1], 'b')
plt.plot(data[(27, 0)]['exact_dynamics_controlled'][:, 2], 'k')
plt.plot(data[(27, 0)]['ro_based_dynamics_controlled'][:, 0], '+r')
plt.plot(data[(27, 0)]['ro_based_dynamics_controlled'][:, 1], 'xb')
plt.plot(data[(27, 0)]['ro_based_dynamics_controlled'][:, 2], 'ok')
plt.subplot(2, 2, 4)
plt.xticks(list(map(lambda x: x / tau, [0, 1, 2, 3, 4, 5, 6, 7])), latex([0, 1, 2, 3, 4, 5, 6, 7]))
plt.ylim(top=1.1, bottom=-1.1)
plt.plot(data[(27, 13)]['exact_dynamics_controlled'][:, 0], 'r')
plt.plot(data[(27, 13)]['exact_dynamics_controlled'][:, 1], 'b')
plt.plot(data[(27, 13)]['exact_dynamics_controlled'][:, 2], 'k')
plt.plot(data[(27, 13)]['ro_based_dynamics_controlled'][:, 0], '+r')
plt.plot(data[(27, 13)]['ro_based_dynamics_controlled'][:, 1], 'xb')
plt.plot(data[(27, 13)]['ro_based_dynamics_controlled'][:, 2], 'ok')
plt.savefig('plots_experiment1/dynamics.pdf')

light_cone = jnp.linspace(2, 27, 50)
light_cone = 2 ** light_cone
plt.figure()
plt.yscale('log')
legend = []
colors = cm.plasma(jnp.linspace(0, 1, plots_num))
for color , (key, value) in zip(colors, sorted(data.items(), key=lambda kv: kv[0])):
    if key[1] == 0:
        eps = value['params']['eps']
        n = value['params']['n']
        legend.append(r'$n={}, \ \epsilon={}$'.format(n, eps))
        plt.plot(value['ro_dim'], 'o', color=color)
legend.append(r'${\rm Light \ cone}$')
plt.plot(light_cone, 'k')
plt.legend(legend, ncol=2, frameon=False, fontsize=13)
plt.xticks(list(map(lambda x: x / tau, [0, 1, 2, 3, 4, 5, 6, 7])), latex([0, 1, 2, 3, 4, 5, 6, 7]))
plt.xlabel(r'${\rm Time}$', fontsize=14)
plt.ylim(top = 8e3)
plt.savefig('plots_experiment1/reduced-order_model_dim.pdf')
