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
    ):
        """
        @param str parameter_name: Name of the parameter
        @param float initial_value: Initial value for the parameter to be used in the optimization
        @param float lower_limit: Lower limit for the parameter
        @param float upper_limit: Upper limit for the parameter
        @param float amplitude_variation: Amplitude variation for the simplex initialization
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

    def set_control_parameters_list(self, map_index):
        """Set the control parameters list. It is used when the"""
        self.control_parameters_list = [map_index + i + 1 for i in range(self.control_parameters_number)]

    def set_parameter(self, optimized_parameter_vector):
        """
        Set the parameter value after checking the constraints
        @param np.array optimized_parameter_vector: The optimized parameter coming from the optimization algorithm
        @return:
        """
        #
        self._set_parameter(optimized_parameter_vector[0])

    def get_parameter(self, optimized_parameter_vector):
        """

        @param optimized_parameter_vector:
        @return:
        """
        self._set_parameter(optimized_parameter_vector[0])
        return self.value

    def _set_parameter(self, parameter):
        """

        @param float parameter:
        @return:
        """
        self.value = self._check_limits(parameter)

    def _check_limits(self, parameter):
        """
        Check if the optimized parameter respect the amplitude limits before sending to the main controls
        @param float parameter: Parameter coming from the optimization
        @return float: Return the parameter after applying the constraints
        """
        #
        a = self.lower_limit
        #
        b = self.upper_limit
        return np.minimum(np.maximum(a, parameter), b)
