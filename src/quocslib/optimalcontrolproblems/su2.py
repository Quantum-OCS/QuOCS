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


# TODO Add the Hamiltonians and spin operators as constant variables, in the python sense obviously
def hamiltonian_d1_d2(ft, delta1=0.0, delta2=0.0):
    sigma_x = _get_sigma_x()
    sigma_z = _get_sigma_z()

    ham_t = delta1 * sigma_z / 2 + (ft + delta2) * sigma_x / 2
    return ham_t


def hamiltonian_d1_d2_2fields(amplitude_t, phase_t, delta1=0.0, delta2=0.0):
    sigma_x = _get_sigma_x()
    sigma_y = _get_sigma_y()
    sigma_z = _get_sigma_z()
    # The controls
    ham_t = delta1 * sigma_z / 2 + amplitude_t * (1 + delta2) * (np.cos(phase_t) * sigma_x + np.sin(phase_t) * sigma_y)
    return ham_t


def _get_sigma_x():
    sigma_x = np.array([[0, 1], [1, 0]], dtype="complex")
    return sigma_x


def _get_sigma_y():
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype="complex")
    return sigma_y


def _get_sigma_z():
    sigma_z = np.array([[1, 0], [0, -1]], dtype="complex")
    return sigma_z
