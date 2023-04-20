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

import numpy as np
import jax.numpy as jnp
from quocslib.optimalcontrolproblems.su2 import *
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.timeevolution.piecewise_integrator_AD import pw_final_evolution_AD
from quocslib.utils.jax_utils import fidelity_funct as fidelity_funct_AD


class IsingModel(AbstractFoM):
    """
    A figure of merit class for optimization of the problem defined by Alastair Marshall via
    https://arxiv.org/abs/2110.06187
    """
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        ################################################################################################################
        # Dynamics variables
        ################################################################################################################
        self.n_qubits = args_dict.setdefault("n_qubits", 5)
        self.J = args_dict.setdefault("J", 1)
        self.g = args_dict.setdefault("g", 2)
        self.n_slices = args_dict.setdefault("n_slices", 100)

        self.H_drift = jnp.asarray(get_static_hamiltonian(self.n_qubits, self.J, self.g))
        self.H_control = jnp.asarray(get_control_hamiltonian(self.n_qubits))
        self.rho_0 = jnp.asarray(get_initial_state(self.n_qubits))
        self.rho_target = jnp.asarray(get_target_state(self.n_qubits))
        self.rho_final = jnp.asarray(np.zeros_like(self.rho_target))

    def get_control_Hamiltonians(self):
        return self.H_control

    def get_drift_Hamiltonian(self):
        return self.H_drift

    def get_target_state(self):
        return self.rho_target

    def get_initial_state(self):
        return self.rho_0

    def get_propagator(self,
                       pulses_list: list = jnp.array,
                       time_grids_list: list = jnp.array,
                       parameters_list: list = jnp.array) -> np.array:
        """ Compute and return the list of propagators """
        # jax.debug.print(pulses_list)
        # jax.debug.print("get_propagator, pulses_list: {}", pulses_list)
        drive = pulses_list[0, :].reshape(1, len(pulses_list[0, :]))
        n_slices = self.n_slices
        time_grid = time_grids_list[0, :]
        dt = time_grid[-1] / len(time_grid)
        # Compute the time evolution
        return pw_final_evolution_AD(drive, self.H_drift, [self.H_control], n_slices, dt, jnp.identity(2 ** self.n_qubits, dtype=np.complex128))

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
        U_final = self.get_propagator(pulses_list=pulses, time_grids_list=timegrids, parameters_list=parameters)
        rho_final = U_final @ self.rho_0 @ U_final.T.conj()
        fidelity = fidelity_funct_AD(rho_final.T, self.rho_target)
        return {"FoM": fidelity}
