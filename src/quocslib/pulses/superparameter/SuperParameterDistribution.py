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
from abc import abstractmethod
import numpy as np


class SuperParameterDistribution:
    basis_vectors_number: int

    def __init__(self, lower_limit: float = 0.0, upper_limit: float = 5, **kwargs):
        # Create the array to store the basis vector super_parameters
        self.w = np.zeros((self.basis_vectors_number, ), dtype="float")
        self.lower_limit_w = lower_limit
        self.upper_limit_w = upper_limit

    @abstractmethod
    def set_random_super_parameter(self):
        """Spend some words here"""
