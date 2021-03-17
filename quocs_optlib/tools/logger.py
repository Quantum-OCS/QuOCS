import logging
import os
import sys


def create_logger(results_path, is_debug=False):
    """Logger creation for console, log file, and debug log file"""
    log_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
    date_format = '%m/%d/%Y %I:%M:%S'
    print_format = '%(levelname)-8s %(name)-12s %(message)s'
    log_filename = os.path.join(results_path, "logging.log")
    log_debug_filename = os.path.join(results_path, "logging_debug.log")

    logger = logging.getLogger("oc_logger")
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
