import os
from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
import numpy as np
import time
from datetime import datetime

max_counter = 50
sleep_time = 0.2
qubit_obj = OneQubit()
fom_path = "fom.txt"
controls_path = "controls.npz"


def main1():
    pulses_list = [np.random.random((100,))]
    time_grids_list = [np.linspace(0.0, 3.0, 100)]
    fom = qubit_obj.get_FoM(pulses=pulses_list, timegrids=time_grids_list, parameters=[])
    print(fom)


def main():

    while read_pulses_file():
        print("Evaluation completed at time {0}".format(datetime.now().time()))
    print("No more pulses after {0} seconds".format(max_counter*sleep_time))


def read_pulses_file() -> bool:
    counter = 0
    controls_filename = "controls.npz"
    is_running = True
    while counter < max_counter:
        if os.path.isfile(controls_filename):
            # Load the controls
            controls = np.load(controls_filename)
            # Remove controls file
            os.remove(controls_filename)
            # Calculate the figure of merit
            # pulses, parameters, timegrids)
            fom = qubit_obj.get_FoM([controls["pulse1"]], [], [controls["time_grid1"]])
            # Return the figure of merit in the fom file
            with open(fom_path, "w+") as fom_file:
                fom_file.write(str(fom))
                fom_file.close()
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
    main()
