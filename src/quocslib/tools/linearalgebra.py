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
from quocslib.tools.randomgenerator import RandomNumberGenerator


def ptrace(rho, dimensions):
    """
    Useful to have this implementation of the partial trace which uses einsums

    TODO implement this in Python again
    """

    return rho


def commutator(A, B):
    return A @ B - B @ A


def gram_schmidt(A):
    """
    Orthonormalize a set of linear independent vectors

    :param A: Square matrix with linear independent vectors
    :return A: Square matrix with orthonormalize vectors
    """
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            u_k = A[:, k]
            A[:, j] -= np.dot(u_k, A[:, j]) * u_k / np.linalg.norm(u_k)**2
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A


def simplex_creation(mean_value: np.array, sigma_variation: np.array, rng: RandomNumberGenerator = None) -> np.array:
    """
    Creation of the simplex

    @return:
    """
    ctrl_par_number = mean_value.shape[0]
    ##################
    # Scale matrix:
    # Explain what the scale matrix means here
    ##################
    # First row
    x0_scale = np.zeros((1, ctrl_par_number))
    # Simplex matrix ( without first row )
    simplex_matrix = np.diag(np.ones_like(sigma_variation))
    # Add random number in the first column
    if rng is None:
        random_array = np.random.rand(ctrl_par_number)
    else:
        random_array = rng.get_random_numbers(ctrl_par_number)
    random_array = random_array.reshape(ctrl_par_number, )
    simplex_matrix[0, :] += np.sqrt(3) * (random_array - 0.5) * 2
    # Orthogonalize set of vectors with gram_schmidt, and rescale with the normalization length
    simplex_matrix_orthonormal = gram_schmidt(simplex_matrix.T)
    # Rescale the vector with the sigma variation
    simplex_matrix_orthogonal_rescaled = simplex_matrix_orthonormal @ np.diag(sigma_variation)
    # Add the first row containing only zeros
    x_t_norm = np.append(x0_scale, simplex_matrix_orthogonal_rescaled, axis=0)
    # Offset matrix
    x_offset = np.outer(np.ones((1, ctrl_par_number + 1)), mean_value)
    # Start simplex matrix
    StartSimplex = x_t_norm + x_offset
    return StartSimplex


if __name__ == "__main__":
    # TODO Move this main script to a test script
    Nc = 4
    ampl_var_1 = 2.0
    ampl_var_2 = 0.7
    f_norm = 1 / np.sqrt(2)
    p_1 = (ampl_var_1 * f_norm) * np.ones(2, )
    p_2 = (ampl_var_2 * f_norm) * np.ones(2, )
    sc_vec = np.append(p_1, p_2)

    x0_scale = np.zeros((1, Nc))
    # Simplex matrix ( without first row )
    simplex_m = np.diag(sc_vec)
    # Add random number
    simplex_m[0, :] += ((sc_vec[0] / 10.0) * (np.random.rand(Nc, ) - 0.5) * 2)

    simplex_m_r = gram_schmidt(simplex_m.T, sc_vec).T
    # Rescale accordingly to amplitude variation
    # x_norm = A_norm.dot(np.diag(sc_vec))
    # Add first row
    x_t_norm = np.append(x0_scale, simplex_m_r, axis=0)

    print(x_t_norm)


def to_sup_op(H):
    """
    Function to convert a Hamiltonian into a Liouvillian
    """
    dim = np.size(H, 1)
    idm = np.eye(dim)
    return np.kron(idm, H) - np.kron(H.T.conj(), idm)


def to_vec(rho):
    """
    Take an input rho vector and flatten it into a column
    """
    return rho.flatten()
