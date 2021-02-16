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

class PythonClassDictionaries:

    def __init__(self, std_dictionary=None):
        if std_dictionary is None:
            python_class_dictionary = {"ProgramType": "TestClass", "PythonModule": "OptimalControlProblems.Rosenbrock",
                                       "PythonClass": "RosenbrockOptimize", "FurtherArgs": {}}
        else:
            python_class_dictionary = std_dictionary

        self.python_class_dictionary = python_class_dictionary

    def get_python_module(self):
        return self.python_class_dictionary["PythonModule"]

    def get_python_class(self):
        return self.python_class_dictionary["PythonClass"]

    def get_arguments_list(self):
        further_args = self.python_class_dictionary["FurtherArgs"]
        arguments_list = []
        for argument in further_args:
            argument_dict = {}
            value = further_args[argument]
            type_value = "string"
            if type(value) is float:
                type_value = "float"
            elif type(value) is int:
                type_value = "int"
            elif type(value) is bool:
                type_value = "bool"
            argument_dict["type"] = type_value
            argument_dict["name"] = argument
            argument_dict["value"] = value
            arguments_list.append(argument_dict)

        return arguments_list

    def set_argument(self, name, value):
        self.python_class_dictionary["FurtherArgs"][name] = value

    def remove_argument(self, name):
        self.python_class_dictionary["FurtherArgs"].pop(name, None)

    def set_module(self, file):
        self.python_class_dictionary["PythonModule"] = file

    def set_class_name(self, class_name):
        self.python_class_dictionary["PythonClass"]=class_name

    def get_dictionary(self):
        return self.python_class_dictionary

    def get_summary_list(self):
        summary_list = []
        summary_list.append("Python Class FoM evaluation")
        summary_list.append("Python Module: " + str(self.python_class_dictionary["PythonModule"]) )
        summary_list.append("Python Class: " + str(self.python_class_dictionary["PythonClass"]))
        summary_list.append("Further args")
        further_args = self.python_class_dictionary["FurtherArgs"]
        for element in further_args:
            summary_list.append(element + " : " + str(further_args[element]))
        return summary_list
