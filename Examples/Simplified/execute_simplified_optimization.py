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
from opt_dict_creation_simplified_optimization import optimization_dictionary


##########################################################################################
### Definition of FoM Class
##########################################################################################


class Example_FoM_Class(AbstractFoM):

    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.some_variable = args_dict.setdefault("some_variable", 5)
        self.stdev = args_dict.setdefault("stdev", 0.1)

    def get_FoM(self,
                pulses: list = [],
                parameters: list = [],
                timegrids: list = []) -> dict:

        # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # here you can do whatever you want!
        fidelity = abs(np.sum(pulses[0]))
        # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        # the returned st.dev is only used if you use re-eval steps
        # it can be calculated also here, but we use the self. value so we can set it during
        # the creation of the FoM object (in the args_dict)
        return {"FoM": fidelity, "std": self.stdev}


##########################################################################################
### Running an optimization
##########################################################################################

# this does not have an effect yet
args_dict = {"is_noisy": True}

FoM_object = Example_FoM_Class(args_dict=args_dict)

# Define Optimizer
optimization_obj = Optimizer(optimization_dictionary, FoM_object)

optimization_obj.execute()
