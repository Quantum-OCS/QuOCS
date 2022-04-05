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
import time
import numpy as np

from quocslib.utils.AbstractDump import AbstractDump


class BestDump(AbstractDump):
    def __init__(self, results_path: str = ".", date_time: str = ".", **kwargs):
        """
        Dumping class for controls and other data which should be the most useful option for most users.
        :param str results_path: Path of the folder of the results
        :param str date_time: String containing the identifier in the form of date and time
        """
        self.best_controls_path = results_path
        self.date_time = date_time

    def dump_controls(self,
                      pulses: list = [],
                      timegrids: list = [],
                      parameters: list = [],
                      is_record: bool = False,
                      **kwargs) -> None:
        """
        Save the controls in the results folder
        :param list pulses: the list containing the pulses that were optimized
        :param list timegrids: the list containing the time grids that were used in the optimization
        :param list parameters: the list containing the parameters that were optimized
        :param bool is_record: information if the current controls are a new record
        """
        if not is_record:
            return

        controls_dict = {}
        pulse_index = 1

        for pulse, time_grid in zip(pulses, timegrids):
            controls_dict["pulse" + str(pulse_index)] = pulse
            controls_dict["time_grid" + str(pulse_index)] = time_grid
            pulse_index += 1

        parameter_index = 1

        for parameter in parameters:
            controls_dict["parameter" + str(parameter_index)] = parameter
            parameter_index += 1

        # Full dictionary
        full_dict = {**controls_dict, **kwargs}

        # Save the file
        controls_path = os.path.join(self.best_controls_path, self.date_time + "_best_controls.npz")
        np.savez(controls_path, **full_dict)

    def other_dumps(self, filename: str = "test.txt", data: np.array = np.array([0.0])):
        """
        Save other results into a txt numpy file
        :param str: filename
        :param np.array: data
        """
        # Create the path
        path = os.path.join(self.best_controls_path, filename)
        # Save the data in a txt file
        np.savetxt(path, data)
