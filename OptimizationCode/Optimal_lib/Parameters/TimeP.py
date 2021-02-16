"""
Class for Times

"""
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


class Timep(OptiPara):

    def __init__(self, objdict):
        super().__init__(objdict)
        self.parameter = objdict["TT"]
        pass

    """
        def _get_json_time(self, pathfile, timeNr):
        time_dict = readjson(pathfile)
        time = time_dict[1]["times"][timeNr]
        return float(time)
    """

