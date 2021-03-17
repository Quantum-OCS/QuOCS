from setuptools import setup, find_packages

requirements = [
    "scipy",
    "numpy"
]

setup(name="quocs_optlib", packages=find_packages(), version="development", install_requires=requirements)
