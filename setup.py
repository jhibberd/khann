"""Utility script to compile artificial neural network C program as python
extension module.

Usage:
    python setup.py build

See:
    http://docs.python.org/2/extending/building.html

"""

from distutils.core import setup, Extension

mod = Extension(
    "ann",
    sources=["src/annmodule.c", "src/ann.c", "src/storage.c"])

setup(
    name="ArtificialNeuralNetwork",
    version="1.0",
    description="Interface with Artificial Neural Network",
    ext_modules=[mod])

