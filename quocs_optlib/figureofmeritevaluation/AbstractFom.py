from abc import ABCMeta, abstractmethod


class AbstractFom(metaclass=ABCMeta):
    """ Just an abstract class for figure of merit evaluation. Just for test"""

    @abstractmethod
    def get_FoM(self, pulses_list, time_grids_list, parameters_list):
        pass
