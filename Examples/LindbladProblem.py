# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.linalg import sqrtm
from quocslib.utils.AbstractFoM import AbstractFoM
from functools import partial
import sys


class TLSProblem(AbstractFoM):
    """
    This class tackles the Lindblad problem for a single qubit driven about the x-axis under decay to the ground state.
    See the example 6 (equations (17) model, (76) time evolution, (96 - 98) specific Liouvillian)
    in https://doi.org/10.1063/1.5115323
    """

    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}
        ################################################################################################################
        # Dynamics variables
        ################################################################################################################
        self.gamma_decay = args_dict.setdefault("gamma_decay", 0.1)
        self.E = args_dict.setdefault("E", 1)
        self.rho_0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
        self.rho_target = jnp.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=jnp.complex64)

        self.rho_0_LFS = convert_rho_to_LFS(self.rho_0)
        self.rho_target_LFS = convert_rho_to_LFS(self.rho_target)

        self.L_const = jnp.array([[0, 0, 0, self.gamma_decay],
                                  [0, -1.0j*self.E+self.gamma_decay/2, 0, 0],
                                  [0, 0, 1.0j*self.E-self.gamma_decay/2, 0],
                                  [0, 0, 0, -self.gamma_decay]],
                                 dtype=jnp.complex64)

        self.L_drive = jnp.array([[0, 1.0j, -1.0j, 0],
                                  [1.0j, 0, 0, -1.0j],
                                  [-1.0j, 0, 0, 1.0j],
                                  [0, -1.0j, 1.0j, 0]],
                                 dtype=jnp.complex64)

        # Let JAX know to jit the following function
        @jax.jit
        def evolve_rho(drive, time_grid):
            rho_evol = pw_evolution_final(self.rho_0_LFS, self.L_const, self.L_drive, drive, time_grid)
            return rho_evol

        self.evolve_rho = evolve_rho

    def get_FoM(self,
                pulses: list = jnp.array,
                parameters: list = jnp.array,
                timegrids: list = jnp.array) -> dict:
        """
        Function to calculate the figure of merit from the pulses, parameters and timegrids.
        :param pulses: jnp.arrays of the pulses to be optimized.
        :param timegrids: jnp.arrays of the timegrids connected to the pulses.
        :param parameters: jnp.array of the parameters to be optimized.
        :return: dict - The figure of merit in a dictionary
        """
        rho_final_LFS = self.evolve_rho(drive=pulses[0], time_grid=timegrids[0])
        fidelity = fidelity_funct_LFS(rho_final_LFS, self.rho_target_LFS)
        return {"FoM": fidelity}


def fidelity_funct(rho_evolved, rho_aim):
    """
    Function to calculate the fidelity between two density matrices.
    :param rho_evolved:
    :param rho_aim:
    :return: fidelity
    """
    return jnp.abs(jnp.trace(sqrtm(sqrtm(rho_evolved) @ rho_aim @ sqrtm(rho_evolved)))) ** 2


def fidelity_funct_LFS(rho_evolved, rho_aim):
    """
    Function to calculate the fidelity between two density matrices in Fock-Liouville space (FLS).
    :param rho_evolved:
    :param rho_aim:
    :return: fidelity
    """
    return jnp.abs(rho_evolved.conj().T @ rho_aim)


def convert_rho_to_LFS(rho):
    """
    Function to convert a density matrix to a Fock-Liouville space (FLS).
    :param rho:
    :return: rho_LFS
    """
    rho_LFS = jnp.array([rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]], dtype=jnp.complex64)
    return rho_LFS


def convert_LFS_to_rho(rho_LFS):
    """
    Function to convert a Fock-Liouville space (FLS) to a density matrix.
    :param rho_LFS:
    :return: rho
    """
    rho = jnp.array([[rho_LFS[0], rho_LFS[1]], [rho_LFS[2], rho_LFS[3]]], dtype=jnp.complex64)
    return rho


# @partial(jax.jit, static_argnames=["rho_0", "L_const", "L_drive"])
@jax.jit
def pw_evolution_final(rho_0, L_const, L_drive, drive, time_grid):
    """

    """
    rho = rho_0
    dt = time_grid[-1] / len(time_grid)

    def body_fun(i, val):
        L_curr = L_const + L_drive * drive[i]
        U_curr = jsp.linalg.expm(L_curr * dt)
        return U_curr @ val

    rho_evol = jax.lax.fori_loop(0, len(time_grid), body_fun, rho)

    return rho_evol
