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
import os


def hamiltonian_d1_d2(drive, delta1=0.0, delta2=0.0):
    """
    The Hamiltonian to use for the OneQubit problem.
    :param drive: drive amplitude about sigma_x
    :param delta1: detuning on the energy levels
    :param delta2: detuning on the drive
    :return: The Hamiltonian
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype="complex")
    sigma_z = np.array([[1, 0], [0, -1]], dtype="complex")

    ham_t = delta1 * sigma_z / 2 + (drive + delta2) * sigma_x / 2
    return ham_t


class OneQubit(AbstractFoM):
    """
    This class implements the one qubit problem as an example FoM class.
    """
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.psi_target = np.asarray(eval(args_dict.setdefault("target_state", "[1.0/np.sqrt(2), -1j/np.sqrt(2)]")),
                                     dtype="complex")
        self.psi_0 = np.asarray(eval(args_dict.setdefault("initial_state", "[1.0, 0.0]")), dtype="complex")

        # two constant detuning values to use in the Hamiltonian
        self.delta1 = args_dict.setdefault("delta1", 0.1)
        self.delta2 = args_dict.setdefault("delta2", 0.1)

        # Noise in the figure of merit
        self.is_noisy = args_dict.setdefault("is_noisy", False)
        self.noise_factor = args_dict.setdefault("noise_factor", 0.05)

        # Drifting FoM
        self.include_drift = args_dict.setdefault("include_drift", False)
        self.linear_drift_val_over_iteration = args_dict.setdefault("linear_drift_val_over_iteration", 0.002)

        # Maximization or minimization
        # Minimization -1.0
        # Maximization 1.0
        self.optimization_factor = args_dict.setdefault("optimization_factor", -1.0)

        self.FoM_list = []
        self.save_path = ""
        self.FoM_save_name = "FoM.txt"

        self.FoM_eval_number = 0

    def save_FoM(self):
        """Saves the FoM list to a file in the save_path directory"""
        np.savetxt(os.path.join(self.save_path, self.FoM_save_name), self.FoM_list)

    def set_save_path(self, save_path: str = ""):
        """Sets the save path for the FoM list"""
        self.save_path = save_path

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        """
        This function calculates the figure of merit for the one qubit problem.
        :param pulses:
        :param parameters:
        :param timegrids:
        :return dict: Dictionary containing the figure of merit and the standard deviation
        """
        # get the first pulse as a drive
        drive = np.asarray(pulses[0])
        if parameters:
            # checks if list is empty, otherwise it sets delta1 to the first parameter
            self.delta1 = parameters[0]
        # set the time grid
        timegrid = np.asarray(timegrids[0])
        # calculate the time step
        dt = timegrid[1] - timegrid[0]
        # calculate the time evolution operator
        U = self._time_evolution(drive, dt, self.delta1, self.delta2)
        # calculate the final state
        psi_f = np.matmul(U, self.psi_0)
        # calculate the infidelity
        infidelity = 1.0 - self._get_fidelity(self.psi_target, psi_f)
        std = 1e-4
        if self.is_noisy:
            # if the is_noisy flag is set, add noise to the infidelity
            noise = (self.noise_factor * 2 * (0.5 - np.random.rand(1)[0]))
            infidelity += noise
            std = self.noise_factor * 0.6827  # one std contains 68.28% of values

        if self.include_drift:
            # if the include_drift flag is set, add a linear drift to the infidelity
            infidelity += self.linear_drift_val_over_iteration * self.FoM_eval_number

        self.FoM_list.append((-1.0) * self.optimization_factor * infidelity)
        self.FoM_eval_number += 1

        return {"FoM": (-1.0) * self.optimization_factor * infidelity, "std": std}

    @staticmethod
    def _time_evolution(drive, dt, delta1, delta2):
        """
        This function calculates the time evolution operator for the one qubit problem.
        :param drive: driving pulse as a list
        :param dt: time step
        :param delta1: detuning on the energy levels
        :param delta2: detuning on the drive
        :return:
        """
        U = np.identity(2)
        for ii in range(drive.size):
            ham_t = hamiltonian_d1_d2(drive[ii], delta1=delta1, delta2=delta2)
            U_temp = U
            U = np.matmul(expm(-1j * ham_t * dt), U_temp)
        return U

    @staticmethod
    def _get_fidelity(psi1, psi2):
        """
        This function calculates the fidelity between two states.
        :param psi1: state 1
        :param psi2: state 2
        :return float: Fidelity between psi1 and psi2
        """
        return np.abs(np.dot(psi1.conj().T, psi2))**2 / (norm(psi1) * norm(psi2))
