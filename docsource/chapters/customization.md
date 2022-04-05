# Customization
The interface structure of QuOCS easily allows for user customization. 

## Add a new algorithm
To add a new algorithm in the QuOCS library, you need to install the software in the editable mode as explained in [installation](#editable-mode).
Then, use the [algorithm template](https://github.com/Quantum-OCS/QuOCS/blob/main/src/quocslib/optimizationalgorithms/AlgorithmTemplate.py) to implement your own algorithm.

*To easily use the name of your new custom algorithm in the optimization dictionary with the `opti_algorithm_name` key, you have to update the `module_name` and `class_name` keys in [this file](https://github.com/Quantum-OCS/QuOCS/blob/main/src/quocslib/utils/map_dictionary.json)*.

You can test your new algorithm, by using the examples in the [tests](https://github.com/Quantum-OCS/QuOCS/tree/main/tests) folder or by creating your own test. 

## Add a new basis
The same procedure applies for the creation of new basis.
A detailed guide for this procedure is also available [here](https://github.com/Quantum-OCS/QuOCS/wiki/Create-new-basis).
