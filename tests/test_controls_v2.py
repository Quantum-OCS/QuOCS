import os

from quocs_optlib.Controls import Controls
from quocs_optlib.tools.inputoutput import readjson
from quocs_optlib.pulses.basis.Fourier import Fourier
from quocs_optlib.pulses.frequency.Uniform import Uniform

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
