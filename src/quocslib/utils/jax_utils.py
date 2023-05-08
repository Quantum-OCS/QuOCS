import jax.numpy as jnp


def fidelity_funct(rho_evolved, rho_aim):
    """
    Fidelity function via the Hilbert-Schmidt inner product
    :param rho_evolved:
    :param rho_aim:
    :return float: State overlap
    """
    return jnp.abs(jnp.trace(rho_evolved.conj().T @ rho_aim))
