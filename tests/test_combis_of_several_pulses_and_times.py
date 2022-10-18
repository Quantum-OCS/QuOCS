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

from quocslib.Optimizer import Optimizer
import numpy as np
from quocslib.utils.AbstractFoM import AbstractFoM
import pytest, sys, os
from quocslib.utils.inputoutput import readjson


##########################################################################################
### Definition of FoM Class
##########################################################################################


class Example_FoM_Class(AbstractFoM):

    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.stdev = args_dict.setdefault("stdev", 0.1)

    def get_FoM(self,
                pulses: list = [],
                parameters: list = [],
                timegrids: list = []) -> dict:

        # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # here you can do whatever you want!
        fidelity = sum([abs(np.sum(pulse)) for pulse in pulses])
        # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        # the returned st.dev is only used if you use re-eval steps
        # it can be calculated also here, but we use the self. value so we can set it during
        # the creation of the FoM object (in the args_dict)
        return {"FoM": fidelity, "std": self.stdev}


##########################################################################################
### Running an optimization
##########################################################################################


def main(optimization_dictionary: dict):
    args_dict = {}

    FoM_object = Example_FoM_Class(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    optimization_obj.execute()


def test_two_pulses_one_time():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "two_pulses_one_time.json"))
    main(optimization_dictionary)


def test_one_pulse_two_times():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "one_pulse_two_times.json"))
    main(optimization_dictionary)


def test_two_pulses_two_times():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "two_pulses_two_times.json"))
    main(optimization_dictionary)

