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
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.timeevolution.piecewise_integrator_AD import pw_final_evolution_AD
from quocslib.utils.jax_utils import fidelity_funct as fidelity_funct_AD


def tensor_together(A):
    """Takes a list of matrices and multiplies them together with the tensor product"""
    res = np.kron(A[0], A[1])
    if len(A) > 2:
        for two in A[2:]:
            res = np.kron(res, two)
    else:
        res = res
    return res


def get_static_hamiltonian(nqu, J, g):
    """
    Get the static Hamiltonian for the Ising model
    :param nqu: Number of qubits
    :param J: Nearest neighbour coupling
    :param g: Next-nearest neighbour coupling
    :return: The Hamiltonian
    """
    dim = 2**nqu
    H0 = np.zeros((dim, dim), dtype=np.complex128)
    i2 = np.eye(2)
    sz = np.array([[1, 0], [0, -1]], dtype="complex")
    for j in range(nqu):
        # set up holding array
        rest = [i2] * nqu
        # set the correct elements to sz
        # check, so we can implement a loop around
        if j == nqu - 1:
            idx1 = j
            idx2 = 0
        else:
            idx1 = j
            idx2 = j + 1
        rest[idx1] = sz
        rest[idx2] = sz
        H0 = H0 - J * tensor_together(rest)

    for j in range(nqu):
        # set up holding array
        rest = [i2] * nqu
        # set the correct elements to sz
        # check, so we can implement a loop around
        if j == nqu - 1:
            idx1 = j
            idx2 = 1
        elif j == nqu - 2:
            idx1 = j
            idx2 = 0
        else:
            idx1 = j
            idx2 = j + 2
        rest[idx1] = sz
        rest[idx2] = sz
        H0 = H0 - g * tensor_together(rest)
    return H0


def get_control_hamiltonian(nqu: int):
    """"
    Get the control Hamiltonian for the Ising model
    :param nqu: Number of qubits
    :return: The Hamiltonian
    """
    dim = 2**nqu
    H_at_t = np.zeros((dim, dim), dtype=np.complex128)
    i2 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], dtype="complex")
    for j in range(nqu):
        # set up holding array
        rest = [i2] * nqu
        # set the correct elements to sz
        # check, so we can implement a loop around
        rest[j] = sx
        H_at_t = H_at_t + tensor_together(rest)
    return H_at_t


def get_initial_state(nqu: int):
    """
    Get the initial state for the Ising model
    :param nqu:
    :return: Initial state density matrix
    """
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    state = [rho0] * nqu
    return tensor_together(state)


def get_target_state(nqu: int):
    """
    Get the target state for the Ising model
    :param nqu:
    :return: Target state density matrix
    """
    rhoT = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    state = [rhoT] * nqu
    return tensor_together(state)


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
        fidelity = fidelity_funct_AD(rho_final, self.rho_target)
        return {"FoM": fidelity}
