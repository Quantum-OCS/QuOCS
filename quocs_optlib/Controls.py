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

from quocs_optlib.parameters.Parameter import Parameter
from quocs_optlib.parameters.TimeParameter import TimeParameter
from quocs_optlib.tools.dynamicimport import dynamic_import


class Controls:
    """
    This is the main class for the optimization quantities, i.e. pulses, parameters, and times.
    All these quantities are defined in this class and can be accessed by calling the modules here.
    """

    def __init__(self, pulses_list, times_list, parameters_list):
        """
        :param pulses_list: Pulses List
        :param times_list: Times List
        :param parameters_list: Parameters List
        """
        # Map index
        map_index = -1
        # Lists and dictionary for the controls
        self.pulse_objs_list = []
        self.parameter_objs_list = []
        self.times_obj_dictionary = {}
        # Number of control objects and control parameters
        self.controls_number = 0
        self.control_parameters_number = 0
        ###############################################
        # Pulses
        ###############################################
        self.pulses_number = len(pulses_list)
        for pulse in pulses_list:
            # Basis attribute inside the dictionary or module and class relative import
            basis = pulse["basis"]
            basis_attribute = dynamic_import(attribute=basis.setdefault("basis_attribute", None),
                                             module_name=basis.setdefault("basis_module", None),
                                             class_name=basis.setdefault("basis_class", None))
            # Create the pulse obj
            self.pulse_objs_list.append(basis_attribute(map_index, pulse))
            # Update the map index for the next control
            map_index = self.pulse_objs_list[-1].last_index
            # Update number control parameters
            self.control_parameters_number += self.pulse_objs_list[-1].control_parameters_number
        ###############################################
        # Parameters
        ###############################################
        self.parameters_number = len(parameters_list)
        for parameter in parameters_list:
            self.parameter_objs_list.append(Parameter(map_index, parameter))
            # Update the map index for the next control
            map_index = self.parameter_objs_list[-1].last_index
            self.control_parameters_number += self.parameter_objs_list[-1].control_parameters_number
        ###############################################
        # Times
        ###############################################
        self.times_number = len(times_list)
        for time in times_list:
            self.times_obj_dictionary[time["time_name"]] = TimeParameter(**time)
            # TODO Implement the time optimization here

    def select_basis(self) -> None:
        """ Initialize the frequency basis """
        for pulse in self.pulse_objs_list:
            pulse.frequency_distribution_obj.set_random_frequencies()

    def get_sigma_variation(self) -> np.array:
        """ Return a vector with the maximum sigma in the parameters choice for the start simplex """
        sigma_variation_coefficients = np.zeros(self.control_parameters_number, dtype="float")
        # Pulses
        for pulse in self.pulse_objs_list:
            sigma_variation_coefficients[pulse.control_parameters_list] = pulse.scale_coefficients
        # Parameters
        for parameter in self.parameter_objs_list:
            sigma_variation_coefficients[parameter.control_parameters_list] = parameter.amplitude_variation
        # Times
        for time_name in self.times_obj_dictionary:
            time = self.times_obj_dictionary[time_name]
            if time.is_optimization:
                sigma_variation_coefficients[time.control_parameters_list] = time.amplitude_variation
        return sigma_variation_coefficients

    def get_mean_value(self) -> np.array:
        """ Return the mean value """
        mean_value_coefficients = np.zeros(self.control_parameters_number, dtype="float")
        # Pulses
        for pulse in self.pulse_objs_list:
            mean_value_coefficients[pulse.control_parameters_list] = pulse.offset_coefficients
        # Parameters
        for parameter in self.parameter_objs_list:
            mean_value_coefficients[parameter.control_parameters_list] = parameter.value
        # Times
        # TODO In case of time optimization return the initial value
        return mean_value_coefficients

    def update_base_controls(self, optimized_parameters_vector: np.array) -> None:
        """ Update the base controls. Only pulses is enough"""
        # Set the times
        for time_name in self.times_obj_dictionary:
            time = self.times_obj_dictionary[time_name]
            if time.is_optimization:
                time.set_parameter(optimized_parameters_vector[time.control_parameters_list])
        # Set the pulses
        for pulse in self.pulse_objs_list:
            time_name = pulse.time_name
            pulse.set_base_pulse(optimized_parameters_vector[pulse.control_parameters_list],
                                 final_time=self.times_obj_dictionary[time_name].get_time())

    def get_controls_lists(self, optimized_parameters_vector: np.array) -> [list, list, list]:
        """
        Set the optimized control parameters and get the controls
        :param np.array optimized_parameters_vector:
        :return: The pulses, time grids, and the parameters in three different lists of numpy arrays.
        """
        pulses_list = []
        time_grids_list = []
        parameters_list = []
        # Set the times
        for time_name in self.times_obj_dictionary:
            time = self.times_obj_dictionary[time_name]
            if time.is_optimization:
                time.set_parameter(optimized_parameters_vector[time.control_parameters_list])
        # Get the pulses and the timegrids
        for pulse in self.pulse_objs_list:
            time_name = pulse.time_name
            pulses_list.append(pulse.get_pulse(optimized_parameters_vector[pulse.control_parameters_list],
                                               final_time=self.times_obj_dictionary[time_name].get_time()))
            time_grids_list.append(pulse.time_grid)
        # Get the parameters
        for parameter in self.parameter_objs_list:
            parameters_list.append(parameter.get_parameter(
                optimized_parameters_vector[parameter.control_parameters_list]))

        return pulses_list, time_grids_list, parameters_list
