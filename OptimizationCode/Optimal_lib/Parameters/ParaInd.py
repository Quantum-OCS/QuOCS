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

from OptimizationCode.Optimal_lib.Parameters.OptiPara import OptiPara
import numpy as np


class ParaInd(OptiPara):

    def __init__(self, objdict):
        super().__init__(objdict)
        # Take para from a json file
        if "JsonGuessPara" in objdict:
            pathfile = objdict["JsonGuessPara"]["pathfile"]
            paraNr = int(objdict["JsonGuessPara"]["paraNr"])
            self.offset = self._get_json_para(pathfile, paraNr)
        else:
            self.offset = objdict["GuessPara"]
        self.parameter = self.offset
        self.ampl_limits = objdict["AmpLimits"]
        self.ampl_var = objdict["AmplVar"]

        pass

    def get_ampl_var(self):
        return self.ampl_var

    def set_par(self, xx):
        self.parameter = self._check_limits(xx)

    def _check_limits(self, xx):
        a = self.ampl_limits[0]
        b = self.ampl_limits[1]
        tmp_x = np.minimum(np.maximum(a, xx), b)
        return tmp_x

    """
    def _get_json_para(self, pathfile, paraNr):
        para_dict = readjson(pathfile)
        para = para_dict[1]["paras"][paraNr]
        return float(para)
    """

