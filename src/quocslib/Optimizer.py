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
from quocslib.handleexit.AbstractHandleExit import AbstractHandleExit
from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib.utils.BestDump import BestDump
from quocslib.utils.AbstractFoM import AbstractFoM


class Optimizer:
    def __init__(self,
                 optimization_dict: dict = None,
                 FoM_object: AbstractFoM = None,
                 comm_signals_list: [list, list, list] = None,
                 handle_exit_obj: AbstractHandleExit = None):
        """
        Write this docstring
        """
        # Handle exit
        if handle_exit_obj is None:
            handle_exit_obj = HandleExit()

        self.interface_job_name = optimization_dict.setdefault("optimization_client_name", "run")
        self.create_logfile = optimization_dict.setdefault("create_logfile", True)
        self.dump_format = optimization_dict.setdefault("dump_format", "npz")
        self.communication_obj = AllInOneCommunication(interface_job_name=self.interface_job_name,
                                                       FoM_obj=FoM_object,
                                                       handle_exit_obj=handle_exit_obj,
                                                       dump_attribute=BestDump,
                                                       comm_signals_list=comm_signals_list,
                                                       create_logfile=self.create_logfile,
                                                       dump_format=self.dump_format)

        self.results_path = self.communication_obj.results_path

        algorithm_dict = optimization_dict['algorithm_settings']
        self.optimizer_attribute = dynamic_import(attribute=algorithm_dict.get("algorithm_attribute", None),
                                                  module_name=algorithm_dict.get("algorithm_module", None),
                                                  class_name=algorithm_dict.get("algorithm_class", None),
                                                  name=algorithm_dict.setdefault("algorithm_name", None),
                                                  class_type='algorithm')

        self.opt_alg_obj: OptimizationAlgorithm = self.optimizer_attribute(optimization_dict=optimization_dict,
                                                                           communication_obj=self.communication_obj,
                                                                           FoM_object=FoM_object)

    def execute(self):
        """ Write this docstring """
        self.opt_alg_obj.begin()
        self.opt_alg_obj.run()
        self.opt_alg_obj.end()

    def get_optimization_algorithm(self):
        """ Return the optimization algorithm object """
        return self.opt_alg_obj
