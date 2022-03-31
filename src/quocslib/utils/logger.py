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


def create_logger(results_path: str, logger_name: str = "oc_logger", is_debug: bool = False):
    """Logger creation for console, log file, and debug log file"""
    log_format = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%Y %I:%M:%S"
    print_format = "%(levelname)-8s %(name)-12s %(message)s"
    log_filename = os.path.join(results_path, "logging.log")
    log_debug_filename = os.path.join(results_path, "logging_debug.log")

    logger = logging.getLogger(logger_name)
    # Remove previous handlers if any
    logger.handlers = []
    # Default level for logger
    logger.setLevel(logging.DEBUG)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(print_format))
    # Log file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    # Add handlers to the logger
    logger.addHandler(file_handler)
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
