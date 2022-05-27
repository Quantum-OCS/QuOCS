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


def fidelity_funct(rho_evolved, rho_aim):
    return np.abs(np.trace(rho_evolved.conj() @ rho_aim))


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
    # get the controls
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
