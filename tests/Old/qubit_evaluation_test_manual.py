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
from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
from quocslib.utils.inputoutput import readjson
import numpy as np
import time
from datetime import datetime

max_counter = 1000
sleep_time = 0.2
qubit_obj = OneQubit()
FoM_path = "FoM.txt"
controls_path = "controls."


def main1():
    pulses_list = [np.random.random((100, ))]
    time_grids_list = [np.linspace(0.0, 3.0, 100)]
    FoM = qubit_obj.get_FoM(pulses=pulses_list, timegrids=time_grids_list, parameters=[])
    print(FoM)


def main(controls_file_extension: str = "txt"):

    while read_pulses_file(controls_file_extension=controls_file_extension):
        print("Evaluation completed at time {0}".format(datetime.now().time()))
    print("No more pulses after {0} seconds".format(max_counter * sleep_time))


def read_pulses_file(controls_file_extension: str = "txt") -> bool:
    counter = 0
    is_running = True
    while counter < max_counter:
        if os.path.isfile(controls_path):
            # Wait before read it
            time.sleep(0.05)
            # Load the controls
            # controls = np.load(controls_path, allow_pickle=True, fix_imports=True, encoding='latin1')
            if controls_file_extension == "txt":
                controls = np.loadtxt(controls_path)
                # Print the controls shape
                # print(controls.shape)
                # Dictionary for the controls
                controls_dict = {"time_grid1": controls[0, :], "pulse1": controls[1, :]}
                controls = controls_dict
            elif controls_file_extension == "json":
                controls = readjson(controls_path)
            else:
                # TODO do something here
                print("{0} extension not recognized".format(controls_file_extension))
                is_running = False
                return is_running
            # Remove controls file
            os.remove(controls_path)
            # Calculate the figure of merit
            # pulses, parameters, timegrids)
            FoM = qubit_obj.get_FoM([controls["pulse1"]], [], [controls["time_grid1"]])
            # Return the figure of merit in the FoM file
            with open(FoM_path, "w+") as FoM_file:
                FoM_file.write(str(FoM["FoM"]))
                FoM_file.close()
            break
        else:
            counter += 1
            time.sleep(sleep_time)
    # Check for exit
    if counter == max_counter:
        is_running = False
    # Return if it is running
    return is_running


if __name__ == "__main__":
    import sys
    args_number = len(sys.argv)
    if args_number == 2:
        controls_file_extension = sys.argv[1]
        print("Searching for {0} ".format(controls_path + controls_file_extension))
        controls_path = controls_path + controls_file_extension
        main(controls_file_extension=controls_file_extension)
    else:
        print("Use the file extension as parameter")
