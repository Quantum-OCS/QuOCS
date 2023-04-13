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
import jax.scipy as jsp


def pw_evolution_AD_old(U_store, drive, A, B, n_slices, dt):
    """Compute the piecewise evolution of a system defined by the
    Hamiltonian H = A + drive * B and store the result in U_store

    :param List[np.matrix] U_store: the storage for all of the computed propagators
    :param np.array drive: an array of dimension n_controls x n_slices that contains the amplitudes of the pulse
    :param np.matrix A: the drift Hamiltonian
    :param List[np.matrix] B: the control Hamiltonians
    :param int n_slices: number of slices
    :param float dt: the duration of each time slice
    :return None: Stores the new propagators so this doesn't return
    """
    K = len(B)
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U_store.at[i].set(jsp.linalg.expm(-1j * dt * H))
    return U_store

# def _cumsum(res, el):
#     """
#     - `res`: The result from the previous loop.
#     - `el`: The current array element.
#     """
#     res = res + el
#     return res, res  # ("carryover", "accumulated")
def pw_evolution_AD(U_store, drive, A, B, n_slices, dt):
    """Compute the piecewise evolution of a system defined by the
    Hamiltonian H = A + drive * B and store the result in U_store

    :param List[np.matrix] U_store: the storage for all of the computed propagators
    :param np.array drive: an array of dimension n_controls x n_slices that contains the amplitudes of the pulse
    :param np.matrix A: the drift Hamiltonian
    :param List[np.matrix] B: the control Hamiltonians
    :param int n_slices: number of slices
    :param float dt: the duration of each time slice
    :return None: Stores the new propagators so this doesn't return
    """
    K = len(B)
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U_store.at[i].set(jsp.linalg.expm(-1j * dt * H))
    return U_store
# from jax import lax
#
#
# def cumsum(res, el):
#     """
#     - `res`: The result from the previous loop.
#     - `el`: The current array element.
#     """
#     res = res + el
#     return res, res  # ("carryover", "accumulated")
#
#
# result_init = 0
# final, result = lax.scan(cumsum, result_init, H)
# result


def pw_final_evolution_AD(drive, A, B, n_slices, dt, u0):
    """Compute the piecewise evolution of a system defined by the
    Hamiltonian H = A + drive * B and concatenate all the propagators

    :param np.array drive: an array of dimension n_controls x n_slices that contains the amplitudes of the pulse
    :param np.matrix A: the drift Hamiltonian
    :param List[np.matrix] B: the control Hamiltonians
    :param int n_slices: number of slices
    :param np.matrix u0: the initial propagator to start from
    :return np.matrix: the final propagator
    """
    K = len(B)
    U = u0
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U = jsp.linalg.expm(-1j * dt * H) @ U
    return U


def pw_final_evolution_AD(drive, A, B, n_slices, dt, U0):
    """Compute the piecewise evolution of a system defined by the
    Hamiltonian H = A + drive * B and concatenate all the propagators

    :param np.array drive: an array of dimension n_controls x n_slices that contains the amplitudes of the pulse
    :param np.matrix A: the drift Hamiltonian
    :param List[np.matrix] B: the control Hamiltonians
    :param int n_slices: number of slices
    :param np.matrix u0: the initial propagator to start from
    :return np.matrix: the final propagator
    """
    U = U0
    def body_fun(i, val):
        U = jsp.linalg.expm(-1.0j * dt * (A + B[0]*drive[0, i]))
        return U @ val
    U = jax.lax.fori_loop(0, n_slices, body_fun, U)
    return U