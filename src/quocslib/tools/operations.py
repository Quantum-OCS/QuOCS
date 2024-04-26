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
import jax.numpy as jnp
from jax import jit

@jit
def state_to_dm(state: jnp.array):
    return jnp.outer(state, jnp.conj(state))

@jit
def _sqrtm(a):
    eig_vals, V = jnp.linalg.eigh(a)
    # eig_vals = jnp.linalg.eigvals(a)
    sqrt_eig_vals = jnp.sqrt(eig_vals)
    V_inv = jnp.linalg.inv(V)
    return jnp.matmul(jnp.matmul(V, jnp.diag(sqrt_eig_vals)), V_inv)

def _displace(a: jnp.ndarray, alpha: complex, offset=0) -> jnp.ndarray:
    """Displacement operator using jax

    Parameters
    ----------
    a : jnp.ndarray
        _description_
    alpha : complex
        _description_
    offset : int, optional
        _description_, by default 0

    Returns
    -------
    jnp.ndarray
        _description_
    """
    a_dag = jnp.transpose(jnp.conj(a))
    D = jsp.linalg.expm(alpha * a_dag - jnp.conj(alpha) * a)
    return D