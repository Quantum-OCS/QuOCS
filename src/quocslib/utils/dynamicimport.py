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

import importlib
import os
from quocslib.utils.inputoutput import readjson

folder = os.path.dirname(os.path.realpath(__file__))
total_dict = readjson(os.path.join(folder, "map_dictionary.json"))


def dynamic_import(attribute=None,
                   module_name: str = None,
                   class_name: str = None,
                   name: str = None,
                   class_type: str = None) -> callable:
    """
    Function for dynamic import.
    :param attribute: The attribute of the class you want to use. It is an optional argument.
    :param module_name: Relative import of the module
    :param class_name: Name of the class inside the module
    :param name: Name of the class in the map_dictionary.json
    :param class_type: Type of class, i.e. algorithm, dsm_settings, basis or superparameter_distribution
    :return: The attribute to use to create the object
    """
    # If the attribute is already given, then just return the attribute
    if attribute is not None:
        return attribute

    elif name is not None:
        map_dict = {}
        try:
            if class_type == 'algorithm':
                map_dict = total_dict['opti_algorithm_map']
            elif class_type == 'dsm_settings':
                map_dict = total_dict['dsm_settings_map']
            elif class_type == 'basis':
                map_dict = total_dict['basis_map']
            elif class_type == 'superparameter_distribution':
                map_dict = total_dict['superparameter_distribution_map']
            elif class_type is None:
                raise Exception('The type of the class is not provided!')

            name_dict = map_dict[name]
            module_name = name_dict["module_name"]
            class_name = name_dict["class_name"]
            attribute = getattr(importlib.import_module(module_name), class_name)
            return attribute

        # TODO Substitute the Exception with a proper Error
        except Exception as ex:
            print("{0}.py module does not exist or {1} is not the class in that module".format(module_name, class_name))
            return None

    elif all([module_name is not None, class_name is not None]):
        try:
            # provide backward - compatibility after renaming
            # module_name = module_name.replace(
            #     "quocslib.pulses.super_parameter.", "quocslib.pulses.superparameter."
            # )
            attribute = getattr(importlib.import_module(module_name), class_name)
            return attribute
        # TODO Substitute the Exception with a proper Error
        except Exception as ex:
            print("{0}.py module does not exist or {1} is not the class in that module".format(module_name, class_name))
            return None
    else:
        print("module_name: {0} and/or class_name: {1} are None".format(module_name, class_name))
        return None
