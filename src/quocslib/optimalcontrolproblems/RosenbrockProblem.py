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

from quocstools.AbstractFom import AbstractFom
from scipy import optimize
import numpy as np


class Rosenbrock(AbstractFom):
    """A figure of merit class for optimization of the Rosenbrock function given an arbitrary
    number of parameters"""

    def __init__(self, args_dict:dict = None):
        pass

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        return {"FoM": optimize.rosen(np.asarray(parameters))}
