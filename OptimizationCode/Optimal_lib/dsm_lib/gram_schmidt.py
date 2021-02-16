# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np


def gram_schmidt(A, n_l):
    # Get the number of vectors.
    n = A.shape[1]
    #print(np.linalg.det(A))
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            u_k = A[:, k]
            A[:, j] -= np.dot(u_k, A[:, j]) * u_k/ np.linalg.norm(u_k)**2
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j]) * n_l[j]
    return A

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
