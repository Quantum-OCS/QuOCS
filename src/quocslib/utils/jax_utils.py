import jax.numpy as jnp
import jax.scipy as jsp


def fidelity_funct(rho_evolved, rho_aim):
    return jnp.abs(jnp.trace(rho_evolved.conj() @ rho_aim))
