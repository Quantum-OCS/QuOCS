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

from quocslib.utils.AbstractFoM import AbstractFoM
from scipy import optimize
import numpy as np


class Rosenbrock(AbstractFoM):
    """A figure of merit class for optimization of the Rosenbrock function given an arbitrary
    number of parameters"""
    def __init__(self, args_dict: dict = None):
        """Initialize useful arguments"""
        # Noise in the figure of merit
        self.is_noisy = args_dict.setdefault("is_noisy", False)
        self.noise_factor = args_dict.setdefault("noise_factor", 0.05)
        self.std_factor = args_dict.setdefault("std_factor", 0.01)

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        FoM = optimize.rosen(np.asarray(parameters))
        std = 0.0
        if self.is_noisy:
            noise = (self.noise_factor * 2 * (0.5 - np.random.rand(1, )[0]))
            FoM += noise
            std = (self.std_factor * np.random.rand(1, )[0])

        return {"FoM": FoM, "std": std}
