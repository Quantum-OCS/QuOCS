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
from jax.scipy.linalg import sqrtm
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.timeevolution.piecewise_integrator_AD import pw_final_evolution_AD


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
        self.rho_final = jnp.asarray(jnp.zeros_like(self.rho_target))

        # Let JAX know to jit the following function
        @jax.jit
        def _pw_evolution_transform(drive, dt):
            """
            A wrapper function for the piecewise evolution function of QuOCS
            :param drive: list of drive pulses
            :param dt: time step
            :return: final unitary propagator
            """
            return pw_final_evolution_AD(drive,
                                         self.H_drift,
                                         jnp.asarray([self.H_control]),
                                         self.n_slices,
                                         dt,
                                         jnp.identity(2 ** self.n_qubits, dtype=np.complex128))

        self._pw_evolution_transform = _pw_evolution_transform

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
                       parameters_list: list = jnp.array) -> jnp.array:
        """
        Function to calculate the propagator from the pulses, parameters and timegrids.
        :param pulses_list:
        :param time_grids_list:
        :param parameters_list:
        :return: final propagator
        """

        drive = pulses_list[0, :].reshape(1, len(pulses_list[0, :]))
        time_grid = time_grids_list[0, :]
        dt = time_grid[-1] / len(time_grid)

        # Compute the time evolution
        propagator = self._pw_evolution_transform(drive, dt)
        return propagator

    def get_FoM(self,
                pulses: list = jnp.array,
                parameters: list = jnp.array,
                timegrids: list = jnp.array) -> dict:
        """
        Function to calculate the figure of merit from the pulses, parameters and timegrids.
        :param pulses: jnp.arrays of the pulses to be optimized.
        :param timegrids: jnp.arrays of the timegrids connected to the pulses.
        :param parameters: jnp.array of the parameters to be optimized.
        :return dict: The figure of merit in a dictionary
        """
        U_final = self.get_propagator(pulses_list=pulses, time_grids_list=timegrids, parameters_list=parameters)
        rho_final = U_final @ self.rho_0 @ U_final.T.conj()
        fidelity = fom_funct(rho_final, self.rho_target)
        return {"FoM": fidelity}


i2 = np.eye(2)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
psi0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
psiT = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def tensor_together(A):
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
        # set the correct elements to sx
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
    # this does not work with complex matrices
    # return jnp.abs(jnp.trace(sqrtm(sqrtm(rho_evolved) @ rho_aim @ sqrtm(rho_evolved))))**2
    return jnp.sqrt(jnp.abs(jnp.trace(rho_evolved.conj().T @ rho_aim)))
