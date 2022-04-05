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

from quocslib.handleexit.HandleExit import HandleExit
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib.utils.BestDump import BestDump
from quocslib.utils.AbstractFoM import AbstractFoM


class Optimizer:
    def __init__(self, optimization_dict: dict = None, FoM_object: AbstractFoM = None):
        """
        Write this docstring
        """
        self.interface_job_name = optimization_dict.setdefault("optimization_client_name", "run")
        self.communication_obj = AllInOneCommunication(interface_job_name=self.interface_job_name,
                                                       FoM_obj=FoM_object,
                                                       handle_exit_obj=HandleExit(),
                                                       dump_attribute=BestDump)

        self.results_path = self.communication_obj.results_path

        algorithm_dict = optimization_dict['algorithm_settings']
        self.optimizer_attribute = dynamic_import(attribute=algorithm_dict.setdefault("algorithm_attribute", None),
                                                  module_name=algorithm_dict.setdefault("algorithm_module", None),
                                                  class_name=algorithm_dict.setdefault("algorithm_class", None),
                                                  name=algorithm_dict.setdefault("algorithm_name", None),
                                                  class_type='algorithm')

        self.opt_alg_obj = self.optimizer_attribute(optimization_dict=optimization_dict,
                                                    communication_obj=self.communication_obj)

    def execute(self):
        """ Write this docstring """
        self.opt_alg_obj.begin()
        self.opt_alg_obj.run()
        self.opt_alg_obj.end()
