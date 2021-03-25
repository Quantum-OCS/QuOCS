from abc import ABCMeta


class AbstractHandleExit(metaclass=ABCMeta):
    """Abstract to handle the program exit"""
    is_user_running: bool = True
