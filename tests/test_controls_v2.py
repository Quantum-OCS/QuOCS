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

import os

from quocslib.Controls import Controls
from quocslib.tools.inputoutput import readjson
from quocslib.pulses.basis.Fourier import Fourier
from quocslib.pulses.frequency.Uniform import Uniform

"""
Test for controls initialization using an external basis and control distribution
"""


def main(controls_dict):
    # Modify the controls with the attribute field
    controls_dict["pulses"][0]["basis"]["basis_attribute"] = Fourier
    controls_dict["pulses"][0]["basis"]["random_frequencies_distribution"]["distribution_attribute"] = Uniform
    # Initialize controls
    controls_obj = Controls(controls_dict["pulses"], controls_dict["times"], controls_dict["parameters"])
    # Set random frequencies
    controls_obj.select_basis()
    # Sigma variation
    print("sigma_variation = {0}".format(controls_obj.get_sigma_variation()))
    # Mean value
    print("mean_value = {0}".format(controls_obj.get_mean_value()))
    # Get control lists
    controls_list = [pulses_list, time_grids_list, parameters_list] = \
        controls_obj.get_controls_lists(controls_obj.get_mean_value())
    for control in controls_list:
        print("Control: {0}".format(control))
    controls_obj.update_base_controls(controls_obj.get_mean_value())
    print("The initialization version 2 is concluded")


if __name__ == '__main__':
    main(readjson(os.path.join(os.getcwd(), "controls_dictionary_v2.json"))[1])
