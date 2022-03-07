import jax.numpy as jnp
from jax import lax, value_and_grad, jit, random
from typing import Callable, Iterable, Tuple, Any
from functools import partial

from simulators.control_utils import init_optimizer_state, run_step

PyTree = Any

@partial(jit, static_argnums=0)
def _opt_epoch(loss_and_grad, epoch_size, state, constant_params, control_signal):
    def iter(i, val):
        total_loss, state, control_signal = val
        loss_value, grad = loss_and_grad(constant_params, control_signal)
        control_signal, state = run_step(state, grad, control_signal)
        total_loss += loss_value
        return total_loss, state, control_signal
    loss_value, state, control_signal = lax.fori_loop(0, epoch_size, iter, (jnp.array(0.), state, control_signal))
    return loss_value / epoch_size, state, control_signal


def optimize(loss_fn: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
             constant_params: PyTree,
             control_signal: jnp.ndarray,
             number_of_epoch: int,
             epoch_size: int,
             learning_rate: float,
             beta1: float = 0.9,
             beta2: float = 0.999,
             eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[This function perform control signal optimization via Riemannian AMSGrad optimizer.]

    Args:
        loss_fn (Callable[[PyTree, jnp.ndarray], jnp.ndarray]): [loss function]
        constant_params (PyTree): [constant params]
        control_signal (complex valued jnp.ndarray of shape (..., n, m)): [control signal]
        number_of_epoch (int): [number of optimization epoches]
        epoch_size (int): [size of optimization epoch]
        learning_rate (float): [learning rate]
        beta1 (float, optional): [beta1 parameter of AMSGrad]. Defaults to 0.9.
        beta2 (float, optional): [beta2 parameter of AMSGrad]. Defaults to 0.999.
        eps (float, optional): [eps parameter of AMSGrad]. Defaults to 1e-8.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [adjasted comtrol signal and "learning curve"]
    """

    state = init_optimizer_state(control_signal, learning_rate, beta1, beta2, eps)
    loss_and_grad = value_and_grad(loss_fn, argnums=1)
    loss_history = [loss_fn(constant_params, control_signal)]
    for epoch_num in range(number_of_epoch):
        loss_value, state, control_signal = _opt_epoch(loss_and_grad, epoch_size, state, constant_params, control_signal)
        loss_history.append(loss_value)
        print("\t Optimization epoch #{} is finished.".format(epoch_num+1))
    return control_signal, jnp.array(loss_history)


def random_isometric(key: jnp.ndarray, shape : Iterable[int]) -> jnp.ndarray:
    """[This function generates a random isometric matrix.]

    Args:
        key (jnp.ndarray): [PRNGKey]
        shape (Iterable[int]): [shape]

    Returns:
        complex valued jnp.ndarray of shape that is argument of the function: [generated matrix]
    """
    u = random.normal(key, (*shape, 2))
    u = u[..., 0] + 1j * u[..., 1]
    u, _ = jnp.linalg.qr(u)
    return u
