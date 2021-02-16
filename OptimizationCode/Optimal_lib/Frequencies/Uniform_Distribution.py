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

from OptimizationCode.Optimal_lib.Frequencies.FreqDistribution import FreqDistribution
import numpy as np


class Uniform_Distribution(FreqDistribution):

    def __init__(self, pulses):
        super().__init__(pulses)

    def _set_random_frequencies(self):
        # TODO Check whether it's possible to move the a,b,k parameter in the main class
        # Number of random frequencies
        k = self.w_k

        # Limits on the frequencies
        a = self.w_a
        b = self.w_b

        # Frequencies list
        ww = []

        # number of bins to be used in the calculation
        k_bins = (b-a)/k

        # generation of random frequencies
        for i in range(k):
            ww.append(a + i*k_bins + k_bins*np.random.rand())

        self.ww = np.asarray(ww)
