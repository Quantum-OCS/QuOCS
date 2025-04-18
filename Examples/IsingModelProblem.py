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
from scipy.linalg import sqrtm
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.timeevolution.piecewise_integrator import pw_evolution
from quocslib.tools.randomgenerator import RandomNumberGenerator
import functools


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

        self.H_drift = get_static_hamiltonian(self.n_qubits, self.J, self.g)
        self.H_control = get_control_hamiltonian(self.n_qubits)
        self.rho_0 = get_initial_state(self.n_qubits)
        self.rho_target = get_target_state(self.n_qubits)
        self.rho_final = np.zeros_like(self.rho_target)
        # allocate a storage array
        self.prop_store = [
            np.zeros_like(self.H_drift) for _ in range(self.n_slices)
        ]
        self.FoM_list = []
        self.rng = 0
        self.g_seed = args_dict.setdefault("g_seed", 0)
        if self.g_seed != 0:
            self.rng = RandomNumberGenerator(seed_number=self.g_seed)
        self.g_variation = args_dict.setdefault("g_variation", 0)
        self.stdev = args_dict.setdefault("stdev", 0.1)

    def get_control_Hamiltonians(self):
        return self.H_control

    def get_drift_Hamiltonian(self):
        # only use "noise" on g if rng seed is set
        if self.rng != 0:
            return get_static_hamiltonian(self.n_qubits, self.J,
                                          self.g + self.g_variation * (0.5 - self.rng.get_random_numbers(1)[0]))
        else:
            return get_static_hamiltonian(self.n_qubits, self.J, self.g)

    def get_target_state(self):
        return self.rho_target

    def get_initial_state(self):
        return self.rho_0

    def get_propagator(self,
                       pulses_list: list = [],
                       time_grids_list: list = [],
                       parameters_list: list = []) -> np.array:
        """
        This function computes the propagator for the given pulses, parameters and time grids.
        :param pulses_list:
        :param time_grids_list:
        :param parameters_list:
        :return: list of propagators
        """

        drive = pulses_list[0].reshape(1, len(pulses_list[0]))
        n_slices = self.n_slices
        time_grid = time_grids_list[0]
        # dt = time_grid[1] - time_grid[0]
        dt = time_grid[-1] / len(time_grid)
        # Compute the time evolution
        self.prop_store = pw_evolution(self.prop_store, drive,
                                       self.get_drift_Hamiltonian(),
                                       [self.H_control], n_slices, dt)
        return self.prop_store

    def get_FoM(self,
                pulses: list = [],
                parameters: list = [],
                timegrids: list = []) -> dict:
        """
        Function to calculate the figure of merit from the pulses, parameters and timegrids.
        :param pulses:
        :param parameters:
        :param timegrids:
        :return dict: The figure of merit in a dictionary
        """
        # Compute the final propagator
        prop_store = self.get_propagator(pulses, timegrids, parameters)
        U_final = functools.reduce(lambda a, b: b @ a, self.prop_store)
        # evolve initial state
        rho_final = U_final @ self.rho_0 @ U_final.T.conj()
        # Calculate the fidelity
        fidelity = fom_funct(rho_final, self.rho_target)
        self.FoM_list.append(fidelity)
        return {"FoM": -fidelity, "std": self.stdev}


# Define some operators
i2 = np.eye(2)
sz = 0.5 * np.matrix([[1, 0], [0, -1]], dtype=np.complex128)
sx = 0.5 * np.matrix([[0, 1], [1, 0]], dtype=np.complex128)
psi0 = np.matrix([[1, 0], [0, 0]], dtype=np.complex128)
psiT = np.matrix([[0, 0], [0, 1]], dtype=np.complex128)


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
    dim = 2**nqu
    H0 = np.zeros((dim, dim), dtype=np.complex128)
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
    dim = 2**nqu
    H_at_t = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(nqu):
        # set up holding array
        rest = [i2] * nqu
        # set the correct elements to sz
        # check, so we can implement a loop around
        rest[j] = sx
        H_at_t = H_at_t + tensor_together(rest)
    return H_at_t


def get_initial_state(nqu: int):
    state = [psi0] * nqu
    return tensor_together(state)


def get_target_state(nqu: int):
    state = [psiT] * nqu
    return tensor_together(state)


def fom_funct(rho_evolved, rho_aim):
    """
    Function to calculate the overlap between two density matrices.
    :param rho_evolved:
    :param rho_aim:
    :return: overlap fidelity
    """
    return np.abs(np.trace(sqrtm(sqrtm(rho_evolved) @ rho_aim @ sqrtm(rho_evolved))))**2
