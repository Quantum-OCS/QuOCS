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
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.timeevolution.piecewise_integrator import pw_evolution
import functools
from scipy.linalg import sqrtm


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


def fidelity_funct(rho_evolved, rho_aim):
    """
    Function to calculate the overlap between two density matrices.
    :param rho_evolved:
    :param rho_aim:
    :return: overlap fidelity
    """
    return np.abs(np.trace(sqrtm(sqrtm(rho_evolved) @ rho_aim @ sqrtm(rho_evolved)))) ** 2


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

        self.FoM_list = []

        self.is_maximization = args_dict.setdefault("is_maximization", False)
        self.FoM_factor = 1
        if self.is_maximization:
            self.FoM_factor = -1

        self.H_drift = get_static_hamiltonian(self.n_qubits, self.J, self.g)
        self.H_control = get_control_hamiltonian(self.n_qubits)
        self.rho_0 = get_initial_state(self.n_qubits)
        self.rho_target = get_target_state(self.n_qubits)
        self.rho_final = np.zeros_like(self.rho_target)
        # allocate memory for the list containing the propagators
        self.prop_store = [np.zeros_like(self.H_drift) for _ in range(self.n_slices)]
        # Check if the propagators are already computed
        self.propagators_are_computed = False

    def get_control_Hamiltonians(self):
        return self.H_control

    def get_drift_Hamiltonian(self):
        return self.H_drift

    def get_target_state(self):
        return self.rho_target

    def get_initial_state(self):
        return self.rho_0

    def get_propagator(self,
                       pulses_list: list = [],
                       time_grids_list: list = [],
                       parameters_list: list = []) -> np.array:
        """
        Compute and return the list of propagators
        :param pulses_list: List of pulses
        :param time_grids_list: List of time grids
        :param parameters_list: List of parameters
        :return: List of propagators"""
        drive = pulses_list[0].reshape(1, len(pulses_list[0]))
        n_slices = self.n_slices
        time_grid = time_grids_list[0]
        # dt = time_grid[1] - time_grid[0]
        dt = time_grid[-1] / len(time_grid)
        # Compute the time evolution
        self.prop_store = pw_evolution(self.prop_store, drive, self.H_drift, [self.H_control], n_slices, dt)
        self.propagators_are_computed = True
        return self.prop_store

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        """
        Compute and return the figure of merit
        :param pulses: List of pulses
        :param parameters: List of parameters
        :param timegrids: List of time grids
        :return dict: Figure of merit in a dictionary
        """
        # Check if the propagator list is computed before compute the final propagator
        if not self.propagators_are_computed:
            self.get_propagator(pulses_list=pulses, time_grids_list=timegrids, parameters_list=parameters)
        self.propagators_are_computed = False
        # Compute the final propagator
        U_final = functools.reduce(lambda a, b: b @ a, self.prop_store)
        # evolve initial state
        rho_final = U_final @ self.rho_0 @ U_final.T.conj()
        # Calculate the fidelity
        fidelity = -1 * self.FoM_factor * fidelity_funct(rho_final.T, self.rho_target)
        # print("FoM: {}, CheckSum: {}".format(fidelity, np.sum(pulses[0])))
        self.FoM_list.append(fidelity)
        return {"FoM": fidelity}
