from quocslib.utils.AbstractFom import AbstractFom
from quocslib.utils.inputoutput import writejsonfile
import numpy as np
import os
import time


class FilesUpdateFom(AbstractFom):
    """ An evaluation method for the figure of merit via file exchange"""

    def __init__(self, controls_folder: str = ".", fom_folder: str = ".", max_time=60*2, **kwargs) -> None:
        """

        :param str controls_folder: Path of the controls folder
        :param str fom_folder: Path of the figure of merit folder
        :param int max_time: Maximum time in arbitrary units. 1 = 0.5 seconds
        :param kwargs: Other parameters
        """
        # Control folder
        self.controls_path = os.path.join(controls_folder, "controls.json")
        # Fom folder
        self.fom_path = os.path.join(fom_folder, "fom.txt")
        # Maximum time to wait for figure oif merit file
        self.max_time = max_time

    def get_FoM(self, pulses: list = [], timegrids: list = [], parameters: list = []) -> dict:
        """
        Write the control in the controls.npz file, read the figure of merit in the fom.txt file
        """
        # If fom.txt file exists remove it
        if os.path.exists(self.fom_path):
            os.remove(self.fom_path)
        # Write the pulses into a file or multiple files
        self._put_controls_user_path(pulses, timegrids, parameters)
        # Read the content of fom.txt file
        return self.read_fom_values()

    def read_fom_values(self) -> dict:
        """
        Read the figure of merit fom.txt file
        :return: dict The figure of merit dictionary
        """
        # Read the fom
        fom = None
        time_counter = 0
        time_counter_max = self.max_time * 2
        # Check if figure of merit file exists
        while not os.path.exists(self.fom_path):
            time_counter += 1
            time.sleep(0.5)
            if time_counter >= time_counter_max:
                return {"FoM": fom, "error_code": -2}
        # Sleep to be sure the file is correctly close
        time.sleep(0.01)
        try:
            with open(self.fom_path, "r") as fom_file:
                fom = float(str(fom_file.readline()).strip())
        except Exception as ex:
            print("Unhandled exception during fom reading: {0}".format(ex.args))
            return {"FoM": fom, "error_code": -3}
        return {"FoM": fom}

    def _put_controls_user_path(self, pulses_list: list, time_grids_list: list, parameters_list: list) -> None:
        """
        Save the controls in the controls.npz file
        :param list pulses_list: List of np.arrays. One np.array for each pulse
        :param list time_grids_list: List of np.arrays. One np.array for each time grid
        :param list parameters_list: List of floats. One float ofr each parameter
        :return:
        """
        controls_dict = {}
        pulse_index = 1
        for pulse, time_grid in zip(pulses_list, time_grids_list):
            controls_dict["pulse" + str(pulse_index)] = pulse
            controls_dict["time_grid" + str(pulse_index)] = time_grid
            pulse_index += 1
        parameter_index = 1
        for parameter in parameters_list:
            controls_dict["parameter" + str(parameter_index)] = parameter
            parameter_index += 1
        writejsonfile(self.controls_path, controls_dict)
        # np.savez(controls_path, **controls_dict)

