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

import logging
import os
import sys
import time


def create_logger(results_path, date_time, create_logfile=True, console_info=True, is_debug=False):
    """
    Create a logger object based on the logging module. >It defines a custom format for the log messages, the date and
    printed messages and the log file name. The logger object is returned and can be used to log messages in the code.

    :param results_path: Path where the log file will be saved
    :param date_time: Date and time of the execution (for the log file name)
    :param create_logfile: Boolean to create the log file or not
    :param console_info: Boolean to show info in the console or not
    :param is_debug: Boolean to activate the debug mode
    :return: logger object
    """
    log_format = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
    date_format = "%d/%m/%Y %H:%M:%S"
    print_format = "%(levelname)-8s %(name)-12s %(message)s"
    log_filename = os.path.join(results_path, date_time + "_logging.log")
    log_debug_filename = os.path.join(results_path, date_time + "_logging_debug.log")

    logger = logging.getLogger("oc_logger")
    # Remove previous handlers if any
    logger.handlers = []
    # Default level for logger
    logger.setLevel(logging.INFO)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if console_info:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter(print_format))
    # Log file handler
    if create_logfile:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        # Add handler for logfile to the logger
        logger.addHandler(file_handler)
    # Add handler for console to the logger
    logger.addHandler(console_handler)
    ################################################
    # Debug
    # In case of debug another object handler is activated
    ################################################
    if is_debug:
        # Log debug file handler
        debug_file_handler = logging.FileHandler(log_debug_filename)
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(logging.Formatter(log_format, date_format))
        #
        logger.addHandler(debug_file_handler)
    return logger
