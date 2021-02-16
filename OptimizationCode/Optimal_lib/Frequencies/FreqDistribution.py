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

class FreqDistribution:

    # Default frequency limits
    w_a = 0.5
    w_b = 25.0
    # Default number of frequencies in the pulse
    w_k = 1
    ww = None

    def __init__(self, pulses):
        # Set General Parameters for frequency
        freq_range = pulses['FreqRange']
        self.w_a = freq_range[0]
        self.w_b = freq_range[1]
        # TODO Handle the frequencies number
        n_freqs = pulses.setdefault('NumberFreq', 1)
        self.w_k = n_freqs

    def get_random_frequencies(self):
        self._set_random_frequencies()
        return self.ww

    def _set_random_frequencies(self): pass
