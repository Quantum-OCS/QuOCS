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

from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.utils.inputoutput import writejsonfile
import numpy as np
import os
import time


class FilesUpdateFoM(AbstractFoM):
    """An evaluation method for the figure of merit via files exchange. The communication object accesses to the
    get_FoM function.
    The get_FoM removes the "FoM.txt" file and creates a json, a txt or a npz file in the controls
    folder designed by the user.
    Finally, the read_FoM_values will wait and check for the figure of merit in the folder designed by the user.
    In case none figure of merit is provided by the user in the limited time defined by the user in the constructor or
    an error occur during the evaluation an error will set in the
    """
    def __init__(self,
                 controls_folder: str = ".",
                 is_splitted: bool = False,
                 file_extension: str = "json",
                 FoM_folder: str = ".",
                 max_time: float = 60 * 2,
                 **kwargs) -> None:
        """

        :param str controls_folder: Path of the controls folder
        :param str FoM_folder: Path of the figure of merit folder
        :param int max_time: Maximum time in arbitrary units. 1 = 0.5 seconds
        :param kwargs: Other parameters
        """
        # Control folder
        self.controls_path = os.path.join(controls_folder, "controls.{0}".format(file_extension))
        # File extension
        self.file_extension = file_extension
        # Split the controls in multiple files
        self.is_splitted = is_splitted
        # FoM folder
        self.FoM_path = os.path.join(FoM_folder, "FoM.txt")
        # Maximum time in seconds to wait for the figure of merit evaluation
        self.max_time = max_time

    def get_FoM(self, pulses: list = [], timegrids: list = [], parameters: list = []) -> dict:
        """
        Write the controls in the controls.npz file, read the figure of merit in the FoM.txt file
        """
        print("Removing the previous {0} file if any".format(self.FoM_path))
        # If FoM.txt file exists remove it
        if os.path.exists(self.FoM_path):
            os.remove(self.FoM_path)
        # Write the pulses into a file or multiple files
        # TODO Multiple files extension
        print("Putting controls in {0}".format(self.controls_path))
        self.put_controls_into_user_path(pulses, timegrids, parameters)
        # Read the content of FoM.txt file
        print("Reading the {0} file ".format(self.FoM_path))
        return self.read_FoM_values()

    def read_FoM_values(self) -> dict:
        """
        Read the figure of merit FoM.txt file
        :return: dict The figure of merit dictionary
        """
        # Read the FoM
        FoM = None
        time_counter = 0
        time_counter_max = self.max_time * 2
        # Check if figure of merit file exists
        while not os.path.exists(self.FoM_path):
            time_counter += 1
            time.sleep(0.5)
            if time_counter >= time_counter_max:
                return {"FoM": FoM, "status_code": -2}
        # Sleep to be sure the file is correctly close
        time.sleep(0.01)
        try:
            with open(self.FoM_path, "r") as FoM_file:
                FoM = float(str(FoM_file.readline()).strip())
        # TODO Add specific exception why the figure of merit is not readable
        except Exception as ex:
            print("Unhandled exception during FoM reading: {0}".format(ex.args))
            return {"FoM": FoM, "status_code": -3}
        return {"FoM": FoM}

    def put_controls_into_user_path(self, pulses_list: list, time_grids_list: list, parameters_list: list) -> None:
        """
        Save the controls in the controls.npz file
        :param list pulses_list: List of np.arrays. One np.array for each pulse
        :param list time_grids_list: List of np.arrays. One np.array for each time grid
        :param list parameters_list: List of floats. One float ofr each parameter
        :return:
        """
        # Choose between save as json file or txt file
        file_extension = self.file_extension
        if file_extension == "txt":
            self._put_controls_txt(pulses_list, time_grids_list, parameters_list)
        elif file_extension == "json":
            self._put_controls_json(pulses_list, time_grids_list, parameters_list)
        else:
            # TODO Return a negative status code
            print("The extension {0} is not recognized".format(file_extension))

    def _put_controls_txt(self, pulses_list: list, time_grids_list: list, parameters_list: list):
        """Ordered the controls like pulse1, time_grid1, pulse2, time_grid2, ... para1, para2 ... and save into a
        txt file.
        """
        with open(self.controls_path, "wb") as controls_file:
            for pulse, time_grid in zip(pulses_list, time_grids_list):
                np.savetxt(controls_file, [pulse], fmt="%f")
                np.savetxt(controls_file, [time_grid], fmt="%f")

            np.savetxt(controls_file, parameters_list, fmt="%f")

    def _put_controls_json(self, pulses_list: list, time_grids_list: list, parameters_list: list):
        """Save into a json file"""
        # Save the pulses and the respective timegrids into a dictionary
        controls_dict = {}
        pulse_index = 1
        # The pulses are saved like pulse#
        # The timegrids are saved like time_grids#
        for pulse, time_grid in zip(pulses_list, time_grids_list):
            controls_dict["pulse" + str(pulse_index)] = pulse
            controls_dict["time_grid" + str(pulse_index)] = time_grid
            pulse_index += 1
        parameter_index = 1
        # The parameters are saved like parameter#
        for parameter in parameters_list:
            controls_dict["parameter" + str(parameter_index)] = parameter
            parameter_index += 1
        writejsonfile(self.controls_path, controls_dict)
