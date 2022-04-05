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

from quocslib.optimalcontrolproblems.su2 import hamiltonian_d1_d2
import numpy as np
from scipy.linalg import expm, norm
from quocslib.utils.AbstractFoM import AbstractFoM
import os


class OneQubit(AbstractFoM):
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.psi_target = np.asarray(eval(args_dict.setdefault("target_state", "[1.0/np.sqrt(2), -1j/np.sqrt(2)]")),
                                     dtype="complex")
        self.psi_0 = np.asarray(eval(args_dict.setdefault("initial_state", "[1.0, 0.0]")), dtype="complex")
        self.delta1 = args_dict.setdefault("delta1", 0.1)
        self.delta2 = args_dict.setdefault("delta2", 0.1)

        # Noise in the figure of merit
        self.is_noisy = args_dict.setdefault("is_noisy", False)
        self.noise_factor = args_dict.setdefault("noise_factor", 0.05)
        self.std_factor = args_dict.setdefault("std_factor", 0.01)

        # Drifting FoM
        self.include_drift = args_dict.setdefault("include_drift", False)
        self.linear_drift_val_over_iterartion = args_dict.setdefault("linear_drift_val_over_iterartion", 0.002)

        # Maximization or minimization
        # Minimization -1.0
        # Maximization 1.0
        self.optimization_factor = args_dict.setdefault("optimization_factor", -1.0)

        self.FoM_list = []
        self.save_path = ""
        self.FoM_save_name = "FoM.txt"

        self.FoM_eval_number = 0

    # def __del__(self):
    #     np.savetxt(os.path.join(self.save_path, self.FoM_save_name), self.FoM_list)

    def save_FoM(self):
        np.savetxt(os.path.join(self.save_path, self.FoM_save_name), self.FoM_list)

    def set_save_path(self, save_path: str = ""):
        self.save_path = save_path

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        f = np.asarray(pulses[0])
        if parameters:
            # checks if list is empty, otherwise it sets delta1 to the first para
            self.delta1 = parameters[0]
        timegrid = np.asarray(timegrids[0])
        dt = timegrid[1] - timegrid[0]
        U = self._time_evolution(f, dt, self.delta1, self.delta2)
        psi_f = np.matmul(U, self.psi_0)
        infidelity = 1.0 - self._get_fidelity(self.psi_target, psi_f)
        std = 1e-4
        if self.is_noisy:
            noise = (self.noise_factor * 2 * (0.5 - np.random.rand(1)[0]))
            infidelity += noise
            std = (self.std_factor * np.random.rand(1)[0])

        if self.include_drift:
            infidelity += self.linear_drift_val_over_iterartion * self.FoM_eval_number

        self.FoM_list.append(np.abs(infidelity))
        self.FoM_eval_number += 1

        return {"FoM": (-1.0) * self.optimization_factor * np.abs(infidelity), "std": std}

    @staticmethod
    def _time_evolution(fc, dt, delta1, delta2):
        U = np.identity(2)
        for ii in range(fc.size - 1):
            ham_t = hamiltonian_d1_d2((fc[ii + 1] + fc[ii]) / 2, delta1=delta1, delta2=delta2)
            U_temp = U
            U = np.matmul(expm(-1j * ham_t * dt), U_temp)
        return U

    @staticmethod
    def _get_fidelity(psi1, psi2):
        return np.abs(np.dot(psi1.conj().T, psi2))**2 / (norm(psi1) * norm(psi2))
