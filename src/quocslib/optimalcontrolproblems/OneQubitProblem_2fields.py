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
from scipy.linalg import expm, norm
from quocslib.utils.AbstractFoM import AbstractFoM


def hamiltonian_d1_d2_2fields(amplitude_t, phase_t, delta1=0.0, delta2=0.0):
    """
    The Hamiltonian to use for the OneQubit problem with 2 driving fields.
    :param amplitude_t: amplitude of the driving field
    :param phase_t: phase between sigma_x and sigma_y
    :param delta1: detuning on the energy levels
    :param delta2: detuning on the drive
    :return: The Hamiltonian
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype="complex")
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype="complex")
    sigma_z = np.array([[1, 0], [0, -1]], dtype="complex")

    ham_t = delta1 * sigma_z / 2 + amplitude_t * (1 + delta2) * (np.cos(phase_t) * sigma_x + np.sin(phase_t) * sigma_y)
    return ham_t


class OneQubit2Fields(AbstractFoM):
    """
    This class implements the one qubit problem with two time-dependent function in the drive
    as an example FoM class.
    """
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.psi_target = np.asarray(
            eval(args_dict.setdefault("target_state", "[1.0/np.sqrt(2), -1j/np.sqrt(2)]")),
            dtype="complex",
        )
        self.psi_0 = np.asarray(eval(args_dict.setdefault("initial_state", "[1.0, 0.0]")), dtype="complex")

        # two constant detuning values to use in the Hamiltonian
        self.delta1 = args_dict.setdefault("delta1", 0.1)
        self.delta2 = args_dict.setdefault("delta2", 0.1)

        # Noise in the figure of merit
        self.is_noisy = args_dict.setdefault("is_noisy", False)
        self.noise_factor = args_dict.setdefault("noise_factor", 0.05)

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        """
        This function calculates the figure of merit for the one qubit problem with two fields.
        :param pulses:
        :param parameters:
        :param timegrids:
        :return dict: The FoM and the standard deviation in a dictionary
        """
        amplitude = np.asarray(pulses[0])
        phase = np.asarray(pulses[1])
        timegrid = np.asarray(timegrids[0])
        dt = timegrid[1] - timegrid[0]
        U = self._time_evolution(amplitude, phase, dt, self.delta1, self.delta2)
        psi_f = np.matmul(U, self.psi_0)
        infidelity = 1.0 - self._get_fidelity(self.psi_target, psi_f)

        std = 0.0
        if self.is_noisy:
            noise = (self.noise_factor * 2 * (0.5 - np.random.rand(1, )[0]))
            infidelity += noise
            std = self.noise_factor * 0.6827  # one std contains 68.28% of values

        return {"FoM": np.abs(infidelity), "std": std}

    @staticmethod
    def _time_evolution(amplitude, phase, dt, delta1=0.0, delta2=0.0):
        """
        This function calculates the time evolution operator for the one qubit problem with two fields.
        :param amplitude:
        :param phase:
        :param dt:
        :param delta1:
        :param delta2:
        :return:
        """
        U = np.identity(2)
        for ii in range(amplitude.size):
            ham_t = hamiltonian_d1_d2_2fields(amplitude[ii],
                                              phase[ii],
                                              delta1,
                                              delta2)
            U_temp = U
            U = np.matmul(expm(-1j * ham_t * dt), U_temp)
        return U

    @staticmethod
    def _get_fidelity(psi1, psi2):
        """
        This function calculates the fidelity between two states.
        :param psi1:
        :param psi2:
        :return float: Fidelity between psi1 and psi2
        """
        return np.abs(np.dot(psi1.conj().T, psi2))**2 / (norm(psi1) * norm(psi2))
