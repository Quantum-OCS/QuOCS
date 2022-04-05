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
import numpy as np

from quocslib.pulses.superparameter.SuperParameterDistribution import (
    SuperParameterDistribution, )
from quocslib.tools.randomgenerator import RandomNumberGenerator


class Uniform(SuperParameterDistribution):
    def __init__(
        self,
        basis_vectors_number,
        super_parameter_distribution_dictionary,
        rng: RandomNumberGenerator = None,
    ):
        """Spend here few words, compulsory arguments for the parent class"""
        self.basis_vectors_number = basis_vectors_number
        super().__init__(**super_parameter_distribution_dictionary)
        self.rng = rng

    def set_random_super_parameter(self):
        """Spend here few words here"""
        # Number of random super_parameters
        k = self.basis_vectors_number
        # Limits on the super_parameters
        a = self.lower_limit_w
        b = self.upper_limit_w
        # number of bins to be used in the calculation
        k_bins = (b - a) / k
        # generation of random super_parameters
        if self.rng is None:
            random_array = np.random.rand(k)
        else:
            random_array = self.rng.get_random_numbers(k)
        for i in range(k):
            self.w[i] = a + i * k_bins + k_bins * random_array[i]
