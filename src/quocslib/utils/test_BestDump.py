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
import pytest
import os
import time
import numpy as np
from quocslib.utils.BestDump import BestDump


def test_dump_controls_record():
    """
    Test for the dumping of controls. Writes the npz file and checks if the data is correct.
    Afterwards it removes the file again.
    """

    folder = os.path.dirname(os.path.realpath(__file__))
    dump_obj = BestDump(folder, "000")

    test_pulse = np.array([1, 2, 3])
    test_timegrid = np.array([4, 5, 6])
    test_params = np.array([7, 8, 9])

    dump_obj.dump_controls([test_pulse], [test_timegrid], [test_params], True)

    outfile_path = os.path.join(dump_obj.best_controls_path, "000_best_controls.npz")
    # print(controls.files)
    with np.load(outfile_path) as controls:
        assert (controls["pulse_1"] == test_pulse).all()
        assert (controls["time_grid_for_pulse_1"] == test_timegrid).all()
        assert (controls["parameter_1"] == test_params).all()
    os.remove(outfile_path)


def test_dump_controls_NO_record():
    """
    Test for the dumping of controls if the is_record flag is set to False. Then checks if in that case the file is not
    written.
    """

    folder = os.path.dirname(os.path.realpath(__file__))
    dump_obj = BestDump(folder, "111")

    test_pulse = np.array([1, 2, 3])
    test_timegrid = np.array([4, 5, 6])
    test_params = np.array([7, 8, 9])

    dump_obj.dump_controls([test_pulse], [test_timegrid], [test_params], False)

    outfile_path = os.path.join(dump_obj.best_controls_path, "111_best_controls.npz")
    file_exists = os.path.exists(outfile_path)
    assert not file_exists


def test_other_dumps():
    """
    Test for the dumping of other data. Writes the txt file and checks if the data is correct. Afterwards it removes
    the file again.
    """

    folder = os.path.dirname(os.path.realpath(__file__))
    dump_obj = BestDump(folder, "000")

    test_pulse = np.array([1, 2, 3])
    outfile_name = "my_test.txt"
    dump_obj.other_dumps(outfile_name, test_pulse)
    outfile_path = os.path.join(dump_obj.best_controls_path, outfile_name)
    test_file_load = np.loadtxt(outfile_path)
    # print(test_file_load)
    assert (test_file_load == test_pulse).all()
    os.remove(outfile_path)
