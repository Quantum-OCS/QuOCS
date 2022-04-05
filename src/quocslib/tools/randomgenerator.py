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
from packaging import version


class RandomNumberGenerator:
    def __init__(self, seed_number: int = None):
        numpy_version = np.__version__
        self.message = ""
        self.rng = None
        self.type = None
        # If no seed is provided use the random numbers without any seed
        if seed_number is None:
            return
        if version.parse(numpy_version) < version.parse("1.16.0"):
            # Try to import the randomgen package
            try:
                import randomgen
                from randomgen import RandomGenerator, MT19937

                self.message = "Import the randomgen library version: {}".format(randomgen.__version__)
                self.rng = RandomGenerator(MT19937(seed=seed_number))
                self.type = "randomgen"
            except ImportError:
                raise ImportError(
                    "Please install randomgen using a compatible version of numpy {0}".format(numpy_version))
        else:
            self.rng = np.random.default_rng(seed_number)
            self.type = "numpy"

    def get_random_numbers(self, n: int):
        """Return an array of random numbers"""
        if self.rng is None:
            return np.random.rand(n)
        else:
            if self.type == "numpy":
                return self.rng.random(n)
            else:
                return self.rng.random_sample(n)


# def get_random_numbers(n: int, rng: np.random.Generator = None):
#     """ Return an array of random numbers between 0 and 1 based on the random generator """
#     if rng is None:
#         return np.random.rand(n)
#     else:
#         return rng.random(n)
