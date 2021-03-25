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

from quocs_optlib.Controls import Controls
from quocs_optlib.tools.inputoutput import readjson
"""
Script to check controls initialization, basis vector (random frequencies), getting sigma variation and
mean value for the start simplex generation.
"""


def main(controls_dict):
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
    print("The initialization is concluded")


if __name__ == '__main__':
    main(readjson(os.path.join(os.getcwd(), "controls_dictionary.json"))[1])
