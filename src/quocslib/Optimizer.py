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
    """
    The Optimizer class is the main class of the optimization. It is responsible for the initialization of the
    optimization algorithm and the communication object. The communication object is responsible for the
    communication between the optimization algorithm and the interface. The optimization algorithm is responsible
    for the optimization itself. The optimizer class is also responsible for the initialization of the
    HandleExit object if it is not configured and passed to the Optimizer object. The HandleExit object is
    responsible for the handling of the exit signals.
    It also has the method execute() to start the optimization and the method get_optimization_algorithm() to
    return the optimization algorithm object.
    """
    def __init__(self,
                 optimization_dict: dict = None,
                 FoM_object: AbstractFoM = None,
                 comm_signals_list: [list, list, list] = None,
                 handle_exit_obj: AbstractHandleExit = None):
        """
        Constructor of the Optimizer class. Initializes the objects for ExitHandling, Communication, Dumping and
        OptimizationAlgorithm.

        :param optimization_dict: Dictionary containing the optimization settings
        :param FoM_object: Figure of Merit object
        :param comm_signals_list: List of signals for the communication object
        :param handle_exit_obj: HandleExit object
        """
        # Handle exit
        if handle_exit_obj is None:
            handle_exit_obj = HandleExit()

        self.interface_job_name = optimization_dict.setdefault("optimization_client_name", "run")
        self.create_logfile = optimization_dict.setdefault("create_logfile", True)
        self.console_info = optimization_dict.setdefault("console_info", True)
        self.dump_format = optimization_dict.setdefault("dump_format", "npz")
        self.optimization_direction = optimization_dict["algorithm_settings"].setdefault("optimization_direction",
                                                                                         "minimization")
        self.communication_obj = AllInOneCommunication(interface_job_name=self.interface_job_name,
                                                       FoM_obj=FoM_object,
                                                       handle_exit_obj=handle_exit_obj,
                                                       dump_attribute=BestDump,
                                                       comm_signals_list=comm_signals_list,
                                                       create_logfile=self.create_logfile,
                                                       console_info=self.console_info,
                                                       dump_format=self.dump_format,
                                                       optimization_direction=self.optimization_direction)

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
        """
        Execute the optimization. It runs the begin(), run() and end() methods of the optimization algorithm.
        """
        self.opt_alg_obj.begin()
        self.opt_alg_obj.run()
        self.opt_alg_obj.end()

    def get_optimization_algorithm(self):
        """
        Returns the optimization algorithm object

        :return: Optimization algorithm object
        """
        return self.opt_alg_obj
