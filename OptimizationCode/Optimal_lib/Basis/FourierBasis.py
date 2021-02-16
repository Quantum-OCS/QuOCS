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

from OptimizationCode.Optimal_lib.Basis.ChoppedBasis import ChoppedBasis


class FourierBasis(ChoppedBasis):

    def __init__(self, zz, pulses):
        super().__init__(zz, pulses)
        self.freq_nr = pulses.setdefault('NumberFreq', 1)
        self.n_paras = 2*self.freq_nr
        list_var = []
        for i in range(self.n_paras):
            list_var.append(zz + i + 1)
        self.list_var = list_var
        self.zz = zz + self.n_paras

    def get_sc_coeff(self):
        ampl_var = self.ampl_var*self.var_scale
        sc_coeff = ampl_var/np.sqrt(2) * np.ones((self.n_paras,))
        self.sc_coeff = sc_coeff
        return self.sc_coeff

    def get_offset_coeff(self):
        offset_coeff = np.zeros((1, self.n_paras))
        #offset_coeff = [0.0 for ii in range(self.n_paras)]
        self.offset_coeff = offset_coeff
        return self.offset_coeff

    #TODO To ensure parallelization xx and timegrid should not be mutables class variables
    # Check if they still are
    def _get_shaped_pulse(self):
        # Total shape
        uu = np.zeros(self.nt)
        # Even number of parameters
        c_paras = int(self.n_paras/2)
        TT = self.total_time
        for ii in range(c_paras):
            uu += self.xx[ii]*np.sin(2*np.pi*self.curr_ww[ii]*self.timegrid/TT) + \
                  self.xx[ii+1]*np.cos(2*np.pi*self.curr_ww[ii]*self.timegrid/TT)
        return uu
