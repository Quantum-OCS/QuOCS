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

from quocslib.optimalcontrolproblems.su2 import hamiltonian_d1_d2_2fields
import numpy as np
from scipy.linalg import expm, norm
from quocslib.utils.AbstractFom import AbstractFom


class OneQubit2Fields(AbstractFom):
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.psi_target = np.asarray(eval(args_dict.setdefault("target_state", "[1.0/np.sqrt(2), -1j/np.sqrt(2)]")), dtype="complex")
        self.psi_0 = np.asarray(eval(args_dict.setdefault("initial_state", "[1.0, 0.0]")), dtype="complex")
        self.delta1 = args_dict.setdefault("delta1", 0.1)
        self.delta2 = args_dict.setdefault("delta2", 0.1)

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        amplitude = np.asarray(pulses[0])
        phase = np.asarray(pulses[1])
        timegrid = np.asarray(timegrids[0])
        dt = timegrid[1] - timegrid[0]
        U = self._time_evolution(amplitude, phase, dt, self.delta1, self.delta2)
        psi_f = np.matmul(U, self.psi_0)
        infidelity = 1.0 - self._get_fidelity(self.psi_target, psi_f)
        return {"FoM": infidelity}

    @staticmethod
    def _time_evolution(amplitude, phase, dt, delta1=0.0, delta2=0.0):
        U = np.identity(2)
        for ii in range(amplitude.size - 1):
            ham_t = hamiltonian_d1_d2_2fields((amplitude[ii + 1] + amplitude[ii]) / 2,
                                              (phase[ii + 1] + phase[ii]) / 2, delta1, delta2)
            U_temp = U
            U = np.matmul(expm(-1j * ham_t * dt), U_temp)
        return U

    @staticmethod
    def _get_fidelity(psi1, psi2):
        return np.abs(np.dot(psi1.conj().T, psi2))**2/(norm(psi1)*norm(psi2))
