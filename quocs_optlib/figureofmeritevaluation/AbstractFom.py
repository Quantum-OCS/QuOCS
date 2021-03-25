from abc import ABCMeta, abstractmethod


class AbstractFom(metaclass=ABCMeta):
    """Abstract class for figure of merit evaluation"""

    @abstractmethod
    def get_FoM(self, pulses_list, time_grids_list, parameters_list) -> dict:
        """Abstract method for figure of merit evaluation. It returns a dictionary with
         the FoM key inside """
