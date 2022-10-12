# Feature List of the possible Settings in the Optimization Dictionary

The optimizer of QuOCS need the settings for the optimization to be provided in the form of a dictionary. This `optimization_dictionary` can either be defined in the python code itself or as a JSON file and be read in with the provided `readjson` function. The settings (keys) in that dictionary are listed and explained here.

## General form of the .json file

Assuming you define the settings in the form of a .json file, the general structure should look like this:

~~~yaml
{
    "optimization_client_name": "Name_of_your_Optimization",
    "algorithm_settings": {...},  # settings related to the algorithm
    "pulses": [{...}, {...}, ...],  # list of pulses and their settings
    "parameters": [{...}, {...}, ...],  # list of parameters and their settings
    "times": [{...}, {...}, ...]  # list of times and their settings
}

~~~


