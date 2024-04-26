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

import jax.numpy as jnp
import jax

def create_mask(trace_over_positions, num_subsystems=5):
    """
    Create a mask for the transpose operation, given the positions of subsystems to trace over in a system of 5 subsystems.
    """
    if any(pos < 0 or pos >= num_subsystems for pos in trace_over_positions):
        raise ValueError('Positions must be within the range of the number of subsystems')

    # Create a permutation of subsystems excluding the ones to be traced over
    remaining_positions = [i for i in range(num_subsystems) if i not in trace_over_positions]

    # Initialize the mask with the remaining positions
    mask = remaining_positions + [pos + num_subsystems for pos in remaining_positions]

    # Add the positions of the subsystems to be traced over at the end
    for pos in trace_over_positions:
        mask.append(pos)
        mask.append(pos + num_subsystems)

    return mask

def jax_partial_trace_transform(subsystem_dims, trace_over_positions):
    """
    Compute the partial trace of a quantum state over specified subsystems using JAX.
    """
    num_subsystems = len(subsystem_dims)
    # mask = create_mask(trace_out_positions, num_subsystems)
    mask = create_mask(trace_over_positions, num_subsystems)

    # Final reshape
    remaining_dims = jnp.array([subsystem_dims[i] for i in range(num_subsystems) if i not in trace_over_positions])
    remaining_dim = jnp.prod(remaining_dims)
    @jax.jit
    def partial_trace(state: jnp.ndarray):
        """ Return the partial trace of the state over the specified subsystems. """
        state_reshaped = jnp.reshape(state, subsystem_dims * 2)
        reordered_state = jnp.transpose(state_reshaped, mask)
        for _ in range(len(trace_over_positions)):
            reordered_state = jnp.trace(reordered_state, axis1=-2, axis2=-1)
        return jnp.reshape(reordered_state, [remaining_dim, remaining_dim])

    return partial_trace
