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

from abc import ABCMeta, abstractmethod
import os


class AbstractDump(metaclass=ABCMeta):
    """Abstract class for dumping data"""
    @abstractmethod
    def __init__(self, results_path: str = ".", **kwargs):
        """
        Abstract method for the constructor dumping classes. Set here the relevant paths
        :param str results_path: Path of the folder of the results
        """

    @abstractmethod
    def dump_controls(self,
                      pulses_list: list = [],
                      time_grids_list: list = [],
                      parameters_list: list = [],
                      **kwargs) -> None:
        """
        Abstract method for dumping the controls.
        :param list pulses_list: List of np.array. Every np.array is a pulse.
        :param list time_grids_list: List of np.array. Every np.array is a time grid at each time grid corresponds
        a pulse.
        :param list parameters_list: List of floats.
        :return:
        """
