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

class PureParametrizationDictionary:

    optimization_algorithm = "Pure Parameters Optimization"

    optimization_name = "Test"
    dsm_name_list = ["NelderMead", "CMAES"]
    dsm_name = "NelderMead"

    iteration_number = 10
    xatol = 1e-14
    frtol = 1e-15

    is_adaptive = True

    parameters_number = 2
    parameters_list = []

    general_initial_value = 0.1
    general_lower_limit = -1.0
    general_upper_limit = 1.0
    generaL_variation = (general_upper_limit - general_lower_limit)/10.0

    summary = ""

    description = ""

    def __init__(self):
        # Initialize the parametrization
        self.parameters_list = []
        for i in range(self.parameters_number):
            self._parameter_initialization(i)

    def get_total_dictionary(self):
        general_settings_dict = self.get_general_settings_dict()
        stopping_criteria = self.get_stopping_criteria_dict()
        parameter_opti_list=self.get_parameter_opti_list()
        options_dict = {"stp_criteria": stopping_criteria,
                        "general_settings": general_settings_dict,
                        "pulses": [], "paras": parameter_opti_list,
                        "times": []
                        }
        comm_dict = {"client_job_name": self.optimization_name}

        purepara_dict = {"opti_dict": {"opti_name": "direct_search_1",
                         "options": options_dict,
                         "comm_dict": comm_dict}}
        return purepara_dict

    def get_parameter_opti_list(self):
        parameter_opti_list = []
        for par in self.parameters_list:
            par_dict ={"Name": par["name"], "AmpLimits": [par["lower_limit"], par["upper_limit"]],
                       "GuessPara": par["initial_value"], "AmplVar": par["variation"]}
            parameter_opti_list.append(par_dict)
        return parameter_opti_list

    def get_summary(self):
        summary_list = []
        summary_list.append("This is a " + self.optimization_algorithm)
        summary_list.append("Optimization name : "+self.optimization_name)
        summary_list.append("Direct search method: "+self.dsm_name)
        summary_list.append("General Settings")
        general_settings_dict = self.get_general_settings_dict()
        for element in general_settings_dict:
            summary_list.append("    " + element + ": "+str(general_settings_dict[element]))
        summary_list.append("Stopping Criteria")
        stopping_criteria_dict = self.get_stopping_criteria_dict()
        for element in stopping_criteria_dict:
            summary_list.append("    " +element + ": "+str(stopping_criteria_dict[element]))
        summary_list.append("Controls")
        summary_list.append("Number of parameters: "+str(self.parameters_number))
        for par in self.parameters_list:
            for element in par:
                summary_list.append("    " +element + ": " + str(par[element]))
        return summary_list

    def get_general_settings_dict(self):
        ## General settings dict
        general_settins_dict = {"is_adaptive": self.is_adaptive}
        return general_settins_dict

    def get_stopping_criteria_dict(self):
        ## Stopping Criteria dict
        stopping_criteria_dict = {"maxiter": self.iteration_number, "xatol": self.xatol,
                                       "frtol": self.frtol}
        return stopping_criteria_dict

    # Getter
    def get_dsm_names(self):
        return self.dsm_name_list

    def get_parameters_number(self):
        return self.parameters_number

    def get_parameters_list(self):
        return self.parameters_list

    def get_iteration_number(self):
        pass

    def set_variation(self, variation, index):
        self.parameters_list[index]["variation"] = variation

    def set_upper_limit(self, upper_limit, index):
        self.parameters_list[index]["upper_limit"] = upper_limit

    def set_lower_limit(self, lower_limit, index):
        self.parameters_list[index]["lower_limit"] = lower_limit

    def set_initial_value(self, initial_value, index):
        self.parameters_list[index]["initial_value"] = initial_value

    def set_parameter_name(self, name, index):
        self.parameters_list[index]["name"] = name

    def set_optimization_name(self, optimization_name):
        self.optimization_name = optimization_name

    def set_iteration_number(self, iteration_number):
        self.iteration_number = iteration_number

    def set_dsm_name(self, dsm_name):
        self.dsm_name = dsm_name

    def set_xatol(self, xatol):
        self.xatol = xatol

    def set_frtol(self, frtol):
        self.frtol = frtol

    def set_parameters_number(self, parameters_number):
        if parameters_number < self.parameters_number:
            # Remove last parameter
            for i in range(parameters_number, self.parameters_number):
                del self.parameters_list[-1]
        else:
            # add new parameters
            for i in range(self.parameters_number, parameters_number):
                self._parameter_initialization(i)

        self.parameters_number=parameters_number

    def remove_parameter_dict(self, parameter_position):
        # Remove the parameter dictionary
        del self.parameters_list[parameter_position-1]
        # Update the parameters number
        self.parameters_number = self.parameters_number - 1

    def _parameter_initialization(self, i):
        par_dict = {"name": "parameter_"+str(i+1), "upper_limit": self.general_upper_limit,
                    "lower_limit": self.general_lower_limit,
                    "initial_value":self.general_initial_value,
                    "variation": self.generaL_variation}
        self.parameters_list.append(par_dict)

    def set_values_to_all_parameters(self, index):
        par_dict = self.parameters_list[index]
        for par in self.parameters_list:
            par["upper_limit"] = par_dict["upper_limit"]
            par["lower_limit"] = par_dict["lower_limit"]
            par["initial_value"] = par_dict["initial_value"]
            par["variation"] = par_dict["variation"]
