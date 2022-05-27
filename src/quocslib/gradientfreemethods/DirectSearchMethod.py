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
from quocslib.stoppingcriteria.StoppingCriteria import StoppingCriteria

import numpy as np


class Wrap:
    def __init__(self, args):
        self.args = args
        self.calls_number = [0]
        self.function = None

    def function_wrapper(self, *wrapper_args):
        self.calls_number[0] += 1
        if len(self.args) == 0:
            return self.function(*wrapper_args)
        else:
            return self.function(*(wrapper_args + self.args))

    def wrap_function(self, function):
        if function is None:
            return self.calls_number, None
        self.function = function

        return self.calls_number, self.function_wrapper


class DirectSearchMethod:
    sc_obj: StoppingCriteria

    def __init__(self):
        # TODO Set the initial wrapper function with the target function and the optional arguments
        # TODO Integrate here the callback function for user interruption
        pass

    def run_dsm(self, routine_call: callable, x0: np.array, **kwargs):
        raise NotImplementedError("The direct search method must implement the run_dsm function")

    @staticmethod
    def _get_wrapper(args, func):
        wrap = Wrap(args)
        func = wrap.wrap_function(func)
        return func
