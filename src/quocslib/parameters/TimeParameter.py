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

from quocslib.parameters.BaseParameter import BaseParameter


class TimeParameter(BaseParameter):
    """Class for the time. In case the user chooses to optimize the time, the code redefines the time in function of an
    optimal parameter alpha:
    t = t_0 + alpha(t_max - t_min)
    where t_0 is the guess time, and t_max and t_min the limits for the optimal time.
    The limits for the optimal parameter alpha are then defined as:
    alpha_in = 0
    alpha_max = (t_max - t_0)/(t_max - t_min)
    alpha_min = (t_min - t_0)/(t_max - t_min)
    alpha_variation = (amplitude_variation)/(t_max - t_min)
    """
    def __init__(
        self,
        map_index=-1,
        time_name="time",
        initial_value=1.0,
        is_optimization=False,
        lower_limit=0.0,
        upper_limit=1.0,
        amplitude_variation=0.1,
    ):
        """

        :param map_index:
        :param str time_name: Name of the time parameter assigned by the user
        :param float initial_value: Initial value or fixed value of the time
        :param bool is_optimization: True if the user wants also to optimize the time
        :param float lower_limit: Lower limit for the time
        :param float upper_limit: Upper limit for the time
        :param float amplitude_variation: Simplex points distance
        """
        # time name
        self.time_name = time_name
        self.is_optimization = is_optimization
        # Initialize the value of the time
        self.time = initial_value
        # Calculate the time variation
        self.time_variation = upper_limit - lower_limit
        if self.is_optimization:
            # Define the optimization coefficient alpha to be used in the optimization. Check the info for details
            max_variation = upper_limit - lower_limit
            alpha_max = (upper_limit - initial_value) / max_variation
            alpha_min = (lower_limit - initial_value) / max_variation
            alpha_in = 0.0
            alpha_variation = amplitude_variation / max_variation
            parameter_name = "alpha_" + time_name
            super().__init__(
                map_index=map_index,
                parameter_name=parameter_name,
                initial_value=alpha_in,
                lower_limit=alpha_min,
                upper_limit=alpha_max,
                amplitude_variation=alpha_variation,
            )
        # else:
        # Otherwise just call the parent constructor
        #    super().__init__(parameter_name=time_name, initial_value=initial_value)

    def get_time(self) -> float:
        """Return the time. In case of time optimization, use the alpha parameter to get the optimized time"""
        if self.is_optimization:
            # value is the optimized parameter
            time = self.time + self.value * self.time_variation
        else:
            time = self.time
        return time
