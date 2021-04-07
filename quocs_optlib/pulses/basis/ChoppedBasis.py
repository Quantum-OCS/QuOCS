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
from quocs_tools.dynamicimport import dynamic_import


class ChoppedBasis:
    """
    General class for chopped basis. All the chopped basis has to inherit this class.
    """
    frequencies_number: int

    def __init__(self, basis: dict = None, **kwargs):
        """

        :param basis:
        :param kwargs:
        """
        super().__init__(**kwargs)
        frequency_distribution_dict = basis["random_frequencies_distribution"]
        # Distribution attribute
        distribution_attribute = dynamic_import(
            attribute=frequency_distribution_dict.setdefault("distribution_attribute", None),
            module_name=frequency_distribution_dict.setdefault("distribution_module", None),
            class_name=frequency_distribution_dict.setdefault("distribution_class", None))
        self.frequency_distribution_obj = distribution_attribute(self.frequencies_number, frequency_distribution_dict)

    # Implement here other modules for Chopped Random Basis
