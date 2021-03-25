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


def gram_schmidt(A, n_l):
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            u_k = A[:, k]
            A[:, j] -= np.dot(u_k, A[:, j]) * u_k / np.linalg.norm(u_k)**2
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j]) * n_l[j]
    return A


def simplex_creation(mean_value: np.array, sigma_variation: np.array) -> np.array:
    """

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
    simplex_matrix = np.diag(sigma_variation)
    # Add random number
    simplex_matrix[0, :] += sigma_variation[0] * np.sqrt(3) * (np.random.rand(ctrl_par_number, ) - 0.5) * 2
    # Orthogonalize set of vectors with gram_schmidt, and rescale with the normalization length
    simplex_matrix_orthogonal_rescaled = gram_schmidt(simplex_matrix.T, sigma_variation).T
    # Add first row
    x_t_norm = np.append(x0_scale, simplex_matrix_orthogonal_rescaled, axis=0)
    # Offset matrix
    x_offset = np.outer(np.ones((1, ctrl_par_number + 1)), mean_value)
    # Start simplex matrix
    StartSimplex = x_t_norm + x_offset
    return StartSimplex


if __name__ == '__main__':
    Nc = 4
    ampl_var_1 = 2.0
    ampl_var_2 = 0.7
    f_norm = 1/np.sqrt(2)
    p_1 = (ampl_var_1*f_norm)*np.ones(2,)
    p_2 = (ampl_var_2 * f_norm)*np.ones(2, )
    sc_vec = np.append(p_1, p_2)

    x0_scale = np.zeros((1, Nc))
    # Simplex matrix ( without first row )
    simplex_m = np.diag(sc_vec)
    # Add random number
    simplex_m[0, :] += (sc_vec[0] / 10.0) * (np.random.rand(Nc, ) - 0.5) * 2

    simplex_m_r = gram_schmidt(simplex_m.T, sc_vec).T
    # Rescale accordingly to amplitude variation
    # x_norm = A_norm.dot(np.diag(sc_vec))
    # Add first row
    x_t_norm = np.append(x0_scale, simplex_m_r, axis=0)

    print(x_t_norm)
