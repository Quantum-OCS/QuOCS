import numpy as np
from scipy.linalg import expm

# can we do conditional import?
# try:
#     import jax.scipy as jsp
# except:
#     raise ImportError


# TODO Shall we merge the two functions in one ???
def pw_evolution(U_store, drive, A, B, n_slices, dt):
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
        U_store[i] = expm(-1j * dt * H)
    return None


def pw_final_evolution(drive, A, B, n_slices, dt, u0):
    """Compute the piecewise evolution of a system defined by the
    Hamiltonian H = A + drive * B and concatenate all the propagators

    :param List[np.matrix] U_store: the storage for all of the computed propagators
    :param np.array drive: an array of dimension n_controls x n_slices that contains the amplitudes of the pulse
    :param np.matrix A: the drift Hamiltonian
    :param List[np.matrix] B: the control Hamiltonians
    :param int n_slices: number of slices
    :param np.matrix u0: the initial density matrix to start from
    :return np.matrix: the final propagator
    """
    K = len(B)
    U = u0
    for i in range(n_slices):
        H = A
        for k in range(K):
            H = H + drive[k, i] * B[k]
        U = expm(-1j * dt * H) @ U
    return U
