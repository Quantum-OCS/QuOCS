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
def fidelity_AD(sqrtm_a, b):
    # Calculate the eigenvalues
    # eig_vals = jnp.linalg.eigvals(jnp.matmul(jnp.matmul(sqrtm_a, b), sqrtm_a))
    # 
    eig_vals = jnp.linalg.eigvals(jnp.dot(jnp.dot(sqrtm_a, b), sqrtm_a))
    # Take the real part
    eig_vals = jnp.real(eig_vals)
    # Return the fidelity
    eig_vals = jnp.where(eig_vals > 0.0, eig_vals, 0.0)
    return jnp.power(jnp.sqrt(eig_vals).sum(), 2.0)

@jit
def fidelity_not_diff_due_to_sqrtm(rho, sigma):
    # Compute the square root of rho
    sqrt_rho = jsp.linalg.sqrtm(rho)

    # Compute the product sqrt_rho sigma sqrt_rho
    product = jnp.dot(jnp.dot(sqrt_rho, sigma), sqrt_rho)

    # Take the square root of the product
    sqrt_product = jsp.linalg.sqrtm(product)

    # Compute the trace of the square root of the product
    fid = jnp.trace(sqrt_product)

    # Take the real part to avoid numerical error leading to small imaginary component
    fid = jnp.real(fid)

    return fid