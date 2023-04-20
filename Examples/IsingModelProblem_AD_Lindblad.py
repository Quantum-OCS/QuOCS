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
from jax.experimental.ode import odeint
from quocslib.utils.AbstractFoM import AbstractFoM
# from quocslib.timeevolution.piecewise_integrator_AD import pw_final_evolution_AD


class IsingModel(AbstractFoM):
    """A figure of merit class for optimization of the problem defined by Alastair Marshall via
    https://arxiv.org/abs/2110.06187"""

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
        self.gamma_dephase = args_dict.setdefault("gamma_dephase", 0.1)

        self.H_drift = jnp.asarray(get_static_hamiltonian(self.n_qubits, self.J, self.g))
        self.H_control = jnp.asarray(get_control_hamiltonian(self.n_qubits))
        self.rho_0 = jnp.asarray(get_initial_state(self.n_qubits))
        self.rho_target = jnp.asarray(get_target_state(self.n_qubits))

        self.Lindbladian = jnp.asarray(get_dephasing_Lindbladian(self.n_qubits, self.gamma_dephase))

    # Let JAX know to jit the following function
    @jax.jit
    def _solve_LME(self, H, time_grid):

        rho_0 = self.rho_0

        def LME(rho, t):
            dRho = -1j * (H(t) @ rho - rho @ H(t).conj().T) \
                   + self.Lindbladian @ rho @ self.Lindbladian.conj().T \
                   - 1/2 * self.Lindbladian @ self.Lindbladian.conj().T @ rho \
                   - 1/2 * rho @ self.Lindbladian @ self.Lindbladian.conj().T
            return dRho

        rho_evo = odeint(LME, rho_0, time_grid)
        print(rho_evo[-1])
        # print(solution.ys[-1])
        return rho_evo[-1]

        # return pw_final_evolution_AD(drive,
        #                              self.H_drift,
        #                              jnp.asarray([self.H_control]),
        #                              self.n_slices,
        #                              dt,
        #                              jnp.identity(2 ** self.n_qubits, dtype=np.complex128))

    # self._pw_evolution_transform = _pw_evolution_transform

    def get_control_Hamiltonians(self):
        return self.H_control

    def get_drift_Hamiltonian(self):
        return self.H_drift

    def get_target_state(self):
        return self.rho_target

    def get_initial_state(self):
        return self.rho_0

    # def get_timedep_von_Neumann_eq(self,
    #                                pulse: jnp.array,
    #                                time_grid: jnp.array):
    #
    #     drive = pulses_list[0, :].reshape(1, len(pulses_list[0, :]))
    #     time_grid = time_grids_list[0, :]
    #     dt = time_grid[-1] / len(time_grid)
    #
    #     # Compute the time evolution
    #     # propagator = pw_final_evolution_AD(drive, self.H_drift, jnp.asarray([self.H_control]), n_slices, dt,
    #     #                                    jnp.identity(2 ** self.n_qubits, dtype=np.complex128))
    #     propagator = self._pw_evolution_transform(drive, dt)
    #     return propagator

    def evolve_rho(self, pulse, time_grid):
        T = time_grid[-1]

        def Hamil(t):
            # if t == T:
            #     ind = self.n_slices - 1
            # else:
            #     ind = int(jnp.floor(t/T*self.n_slices))

            ind = jnp.floor(t / T * (self.n_slices-1)).astype(int)
            # print(ind)

            return self.H_drift[ind] + self.H_control[ind] * pulse[ind]

        rho_fin = self._solve_LME(Hamil, time_grid)

        return rho_fin

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
        rho_final = self.evolve_rho(pulse=pulses[0], time_grid=timegrids[0])
        fidelity = fidelity_funct(rho_final, self.rho_target)
        return {"FoM": fidelity}


i2 = np.eye(2)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
psi0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
psiT = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def get_dephasing_Lindbladian(nqu, gamma):
    dim = 2**nqu
    L0 = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(nqu):
        # set up holding array
        rest = [i2] * nqu
        # set the correct elements to sz
        # check, so we can implement a loop around
        rest[j] = sz
        L0 += np.sqrt(gamma) * tensor_together(rest)
    return L0


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


def fidelity_funct(rho_evolved, rho_aim):
    return jnp.abs(jnp.trace(rho_evolved.conj() @ rho_aim))


muh = IsingModel()
muh.evolve_rho(pulse=jnp.array([1, 1, 1, 1]), time_grid=jnp.array([0, 1, 2, 3]))