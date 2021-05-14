# QuOCS: The optimization library
QuOCS (Quantum Optimal Control Suite) is a python software package for theoretical and experimental optimizations.
You can find here the core of the optimization algorithms used in the QuOCS - suite.
## Create a virtual environment
Create a virtual environment:
```bash
python3 -m venv ../quocslib
```
Activate your virtual environment:
```bash
source ../quocslib/bin/activate
```
Install basic packages
```bash
python -m pip install --upgrade pip setuptools wheel
```
Install the QuOCS-tools
```bash
git clone git@github.com:Quantum-OCS/QuOCS-tools.git
cd QuOCS-tools
python -m pip install -e .
```
## Installation
Install quocslib in your virtual environment
```bash
python -m pip install -e .
```

## Tests
Now you are able to use the tests scripts in the tests folder
:)

