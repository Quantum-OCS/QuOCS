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
from quocslib.pulses.BasePulse import BasePulse


class PiecewiseBasis(BasePulse):
    amplitude_variation: float
    optimized_control_parameters: np.ndarray
    optimized_super_parameters: np.ndarray
    time_grid: np.ndarray

    def __init__(self, map_index, pulse_dictionary: dict, rng: RandomNumberGenerator = None):
        """

        :param int map_index: Index number to use to get the control parameters for the Fourier basis
        :param dict pulse_dictionary: The dictionary of the pulse defined here. Only the basis dictionary is used btw

        :param dict pulse_dictionary: Should contain the basis_dict under the key "basis" and should contain the number of bins and time spacing under "n_bins" and "dt"
        """
        # basis_dict = pulse_dictionary["basis"]
        # Constructor of the parent class, i.e. Chopped Basis
        self.control_parameters_number = pulse_dictionary["bins_number"]
        super().__init__(map_index=map_index, rng=rng, **pulse_dictionary)
        # self.control_parameters_number = 1
        # super().__init__(**kwargs)
        # basis_attribute = dynamic_import(attribute=basis.setdefault("basis_attribute", None),
        #                                     module_name=basis.setdefault("basis_module", None),
        #                                     class_name=basis.setdefault("basis_class", None))

        #################
        # Basis dependent settings
        #################

    def setdefault(a, b, c):
        class Skipper:
            def __init__(self):
                self.last_index = 0

        return lambda x, y: Skipper()

    # TODO check if this function is really called by GRAPE
    def _get_shaped_pulse(self) -> np.array:
        """Definition of the pulse parametrization. It is called at every function evaluation to build the pulse"""
        #################
        # Standard Basis Settings: amplitude limits, amplitude variation for the simplex,
        # distribution of super parameters, etc ...
        ################
        pulse = self.pulse_amplitudes
        # in the case of a piecewise basis the pulse is defined by the amplitudes

        return pulse
