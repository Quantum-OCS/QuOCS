# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np

from quocs_optlib.pulses.frequency.FrequencyDistribution import FrequencyDistribution


class Uniform(FrequencyDistribution):

    def __init__(self, basis_vectors_number, frequency_distribution_dictionary):
        """Spend here few words, compulsory arguments for the parent class"""
        self.basis_vectors_number = basis_vectors_number
        super().__init__(**frequency_distribution_dictionary)
        self.lower_limit_w = frequency_distribution_dictionary["lower_limit"]
        self.upper_limit_w = frequency_distribution_dictionary["upper_limit"]

    def set_random_frequencies(self):
        """Spend here few words here"""
        # Number of random frequencies
        k = self.basis_vectors_number
        # Limits on the frequencies
        a = self.lower_limit_w
        b = self.upper_limit_w
        # number of bins to be used in the calculation
        k_bins = (b-a)/k
        # generation of random frequencies
        for i in range(k):
            self.w[i] = a + i*k_bins + k_bins*np.random.rand()
