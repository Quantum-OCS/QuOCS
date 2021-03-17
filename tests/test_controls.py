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
