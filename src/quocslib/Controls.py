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

from quocslib.pulses.basis.ChoppedBasis import ChoppedBasis
from quocslib.parameters.Parameter import Parameter
from quocslib.parameters.TimeParameter import TimeParameter
from quocslib.tools.randomgenerator import RandomNumberGenerator
from quocslib.utils.dynamicimport import dynamic_import


class Controls:
    """
    This is the main class for the optimization quantities, i.e. pulses, parameters, and times.
    All these quantities are defined in this class and can be accessed by calling the modules here.
    """
    def __init__(self, pulses_list, times_list, parameters_list, rng: RandomNumberGenerator = None):
        """
        Constructor of the general class containing all the controls used during the optimization

        :param pulses_list: List containing the settings for each pulse
        :param times_list: List containing the settings for each time
        :param parameters_list: List containing the setting for each parameter
        """
        # Map index
        map_index = -1
        # Lists and dictionary for the controls
        self.pulse_objs_list = []
        self.parameter_objs_list = []
        self.times_obj_dictionary = {}
        ###############################################
        # Pulses
        ###############################################
        self.pulses_number = len(pulses_list)
        for pulse in pulses_list:
            # Basis attribute inside the dictionary or module and class relative import
            basis = pulse["basis"]
            basis_attribute = dynamic_import(attribute=basis.setdefault("basis_attribute", None),
                                             module_name=basis.setdefault("basis_module", None),
                                             class_name=basis.setdefault("basis_class", None),
                                             name=basis.setdefault("basis_name", None),
                                             class_type='basis')
            # Create the pulse obj
            self.pulse_objs_list.append(basis_attribute(map_index, pulse, rng=rng))
            # Update the map index for the next control
            map_index = self.pulse_objs_list[-1].last_index
        ###############################################
        # Parameters
        ###############################################
        self.parameters_number = len(parameters_list)
        for parameter in parameters_list:
            self.parameter_objs_list.append(Parameter(map_index, parameter))
            # Update the map index for the next control
            map_index = self.parameter_objs_list[-1].last_index
        ###############################################
        # Times
        ###############################################
        self.times_number = len(times_list)
        for time in times_list:
            self.times_obj_dictionary[time["time_name"]] = TimeParameter(**time)
            # TODO Implement the time optimization here

    def get_control_parameters_number(self) -> int:
        """Return the control parameter number"""
        ###############################################
        # Pulses
        ###############################################
        control_parameters_number = 0
        for pulse_obj in self.pulse_objs_list:
            # Update number control parameters
            control_parameters_number += pulse_obj.control_parameters_number
        ###############################################
        # Parameters
        ###############################################
        for parameter_obj in self.parameter_objs_list:
            # Update number control parameters
            control_parameters_number += parameter_obj.control_parameters_number
        ###############################################
        # Times
        ###############################################
        # TODO Implement the time optimization here
        # Return the control parameters number
        return control_parameters_number

    def select_basis(self) -> None:
        """Initialize the superparameter basis"""
        for pulse in self.pulse_objs_list:
            pulse.super_parameter_distribution_obj.set_random_super_parameter()
            # Update the base pulse parameters and functions
            pulse.update_chopped_basis()
            # Update the control parameter indexes
            self._update_control_parameter_indexes()

    def _update_control_parameter_indexes(self) -> None:
        """Update the control parameter indexes"""
        # Map index
        map_index = -1
        ###############################################
        # Pulses
        ###############################################
        for pulse_obj in self.pulse_objs_list:
            pulse_obj.set_control_parameters_list(map_index)
            # Update the map index for the next control
            map_index = pulse_obj.last_index
        ###############################################
        # Parameters
        ###############################################
        for parameter_obj in self.parameter_objs_list:
            parameter_obj.set_control_parameters_list(map_index)
            # Update the map index for the next control
            map_index = parameter_obj.last_index
        ###############################################
        # Times
        ###############################################
        # TODO Implement the time optimization here
        # Return the control parameters number

    def get_random_super_parameter(self) -> np.array:
        """Return list with dcrab current super_parameters"""
        super_parameter_list = []
        for pulse in self.pulse_objs_list:
            if isinstance(pulse, ChoppedBasis):
                super_parameter_list.append(pulse.super_parameter_distribution_obj.w)
        super_parameter_array = np.asarray(super_parameter_list)
        return super_parameter_array

    def get_sigma_variation(self) -> np.array:
        """Return a numpy array with the maximum sigma in the parameters choice for the start simplex

        :return np.array
        """
        sigma_variation_coefficients = np.zeros(self.get_control_parameters_number(), dtype="float")
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
        """Return a numpy array the mean value

        :return np.array:
        """
        mean_value_coefficients = np.zeros(self.get_control_parameters_number(), dtype="float")
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
        """Update the base controls. Only pulses is enough"""
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
        # Set the parameters
        for parameter in self.parameter_objs_list:
            parameter.set_parameter(optimized_parameters_vector[parameter.control_parameters_list])

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
            pulses_list.append(
                pulse.get_pulse(optimized_parameters_vector[pulse.control_parameters_list],
                                final_time=self.times_obj_dictionary[time_name].get_time()))
            time_grids_list.append(pulse.time_grid)
        # Get the parameters
        for parameter in self.parameter_objs_list:
            parameters_list.append(
                parameter.get_parameter(optimized_parameters_vector[parameter.control_parameters_list]))

        return pulses_list, time_grids_list, parameters_list
