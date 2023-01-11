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
import json

from quocslib.utils.AbstractDump import AbstractDump


class NumpyEncoder(json.JSONEncoder):
    """
    Class to convert numpy arrays to lists for json dumping
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BestDump(AbstractDump):
    def __init__(self, results_path: str = ".", date_time: str = ".", dump_format: str = "npz", **kwargs):
        """
        Dumping class for controls and other data which should be the most useful option for most users.
        :param str results_path: Path of the folder of the results
        :param str date_time: String containing the identifier in the form of date and time
        """
        self.best_controls_path = results_path
        self.results_path = results_path
        self.date_time = date_time
        self.dump_format = dump_format

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
        pulse_index = 0
        pulse_names = []
        if "pulse_names" in {**kwargs}:
            pulse_names = {**kwargs}["pulse_names"]
        else:
            pulse_names = ["pulse_{}".format(index+1) for index in range(len(pulses))]

        # time_names = []
        # if "time_names" in {**kwargs}:
        #     time_names = {**kwargs}["time_names"]
        # else:
        #     time_names = ["time_grid_{}".format(index + 1) for index in range(len(timegrids))]

        for pulse, time_grid in zip(pulses, timegrids):
            pulse_name = pulse_names[pulse_index]
            if pulse_name in controls_dict:
                pulse_name = pulse_name + str(pulse_index + 1)
            time_name = "time_grid_for_" + pulse_name
            controls_dict[pulse_name] = pulse
            controls_dict[time_name] = time_grid
            pulse_index += 1

        parameter_index = 0
        parameter_names = []
        if "parameter_names" in {**kwargs}:
            parameter_names = {**kwargs}["parameter_names"]
        else:
            parameter_names = ["parameter_{}".format(index+1) for index in range(len(parameters))]

        for parameter in parameters:
            param_name = parameter_names[parameter_index]
            if param_name in controls_dict:
                param_name = param_name + str(parameter_index + 1)
            controls_dict[param_name] = parameter
            parameter_index += 1

        # Full dictionary
        full_dict = {**controls_dict, **kwargs}

        # Save the file
        if self.dump_format == "json":
            self.dump_dict("best_controls", full_dict)
        else:
            controls_path = os.path.join(self.results_path, self.date_time + "_best_controls.npz")
            np.savez(controls_path, **full_dict)

        # if "iteration_number" in full_dict:
        #     iteration_path = os.path.join(self.results_path, self.date_time + "_funct_eval_of_best_controls.txt")
        #     with open(iteration_path, 'w') as f:
        #         f.write(str(full_dict["iteration_number"]))

    def other_dumps(self, filename: str = "test.txt", data: np.array = np.array([0.0])):
        """
        Save other results into a txt numpy file
        :param str: filename
        :param np.array: data
        """
        # Create the path
        path = os.path.join(self.results_path, filename)
        # Save the data in a txt file
        np.savetxt(path, data)

    def dump_dict(self, data_file_name: str = "unknown_data_dict", data_dict=None):
        """
        Save a dictionary to a file
        :param data_file_name:
        :param data_dict:
        :return:
        """
        if data_dict is None:
            data_dict = {}
        data_dict_path = os.path.join(self.results_path, self.date_time + "_" + data_file_name+'.json')
        with open(data_dict_path, 'w') as convert_file:
            convert_file.write(json.dumps(data_dict, indent=4, cls=NumpyEncoder))

