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


class BaseParameter:
    """
    Parameter Class where every parameter to be optimized is defined. It is used as a parent class of the Time class in
    order to optimize the time of the pulse whenever the user specified it in the configuration class.
    """
    def __init__(
        self,
        map_index=-1,
        parameter_name="parameter",
        initial_value=1.0,
        lower_limit=0.1,
        upper_limit=1.1,
        amplitude_variation=0.1,
        is_AD: bool = False,
    ):
        """
        Constructor of the BaseParameter class. It is used to define the parameter to be optimized.

        :param int map_index: Index of the parameter in the map of all parameters
        :param str parameter_name: Name of the parameter
        :param float initial_value: Initial value for the parameter to be used in the optimization
        :param float lower_limit: Lower limit for the parameter
        :param float upper_limit: Upper limit for the parameter
        :param float amplitude_variation: Amplitude variation for the simplex initialization
        """
        # Parameter name
        self.parameter_name = parameter_name
        # Initial value for the optimization
        self.value = initial_value
        # Upper limit
        self.upper_limit = upper_limit
        # Lower limit
        self.lower_limit = lower_limit
        # Amplitude variation
        self.amplitude_variation = amplitude_variation
        # The parameters control number is ever equal to 1
        self.control_parameters_number = 1
        # Create the control parameters list to get the optimized parameter from the optimized vector
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]
        # Update the map_index number for the next pulse
        self.last_index = self.control_parameters_list[-1]
        if is_AD:
            self._set_AD_functions()
        else:
            self._set_functions()

    def _set_AD_functions(self):
        """ Sets jax functions in case automatic differentiation is used """
        import jax
        # self.debug_print = jax.debug.print
        import jax.numpy as jnp
        self._maximum = jnp.maximum
        self._minimum = jnp.minimum
        # Array tipe
        self.array_type = jnp.ndarray

    def _set_functions(self):
        """ Sets standard numpy functions in case automatic differentiation is not used """
        self._maximum = np.maximum
        self._minimum = np.minimum
        # Array tipe
        self.array_type = np.ndarray

    def set_control_parameters_list(self, map_index):
        """Updates the control parameters list."""
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]

    def set_parameter(self, optimized_parameter_vector):
        """
        Sets the parameter value after checking the constraints.

        :param np.array optimized_parameter_vector: The optimized parameter coming from the optimization algorithm
        """
        self._set_parameter(optimized_parameter_vector[0])

    def get_parameter(self, optimized_parameter_vector):
        """
        Gets the parameter value after checking the constraints.

        :param optimized_parameter_vector:
        :return float: The parameter value
        """
        self._set_parameter(optimized_parameter_vector[0])
        return self.value

    def _set_parameter(self, parameter):
        """
        Sets the parameter value after checking the constraints.

        :param float parameter:
        """
        self.value = self._check_limits(parameter)

    def _check_limits(self, parameter):
        """
        Check if the optimized parameter respect the amplitude limits before sending to the main controls.
        If it exceeds one of the limits it is set to the limit value.

        :param float parameter: Parameter coming from the optimization
        :return float: Parameter value after applying the constraints
        """
        a = self.lower_limit
        b = self.upper_limit
        return self._minimum(self._maximum(a, parameter), b)
