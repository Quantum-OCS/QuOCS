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


def dynamic_import(attribute=None, module_name: str = None, class_name: str = None) -> callable:
    """
    Function for dynamic import.
    :param attribute: The attribute of the class you want to use. It is an optional argument.
    :param module_name: Relative import of the module
    :param class_name: Name of the class inside the module
    :return: The attribute to use to create the object
    """
    # If the attribute is already given, then just return the attribute
    if attribute is not None:
        return attribute
    # Get the attribute
    import_conditions = [module_name is not None, class_name is not None]
    if all(import_conditions):
        try:
            attribute = getattr(importlib.import_module(module_name), class_name)
        except Exception as ex:
            print("{0}.py module does not exist or {1} is not the class in that module".format(module_name, class_name))
            return None
    else:
        print("module_name: {0} and/or class_name: {1} are None".format(module_name, class_name))
    return attribute
