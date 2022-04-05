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


class AbstractFoM(metaclass=ABCMeta):
    """Abstract class for figure of merit evaluation"""
    @abstractmethod
    def get_FoM(self, pulses_list: list = [], time_grids_list: list = [], parameters_list: list = []) -> dict:
        """
        Abstract method for figure of merit evaluation. It returns a dictionary with
         the FoM key inside
        :param list pulses_list: List of np.array. Every np.array is a pulse.
        :param list time_grids_list: List of np.array. Every np.array is a time grid at each time grid corresponds
        to a pulse.
        :param list parameters_list: List of floats.
        :return dict: The dictionary must contain at least the "FoM" key with the figure of merit float. Other possible
        keys can provide useful information for errors or other.
        """
