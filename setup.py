"""Utility script to compile artificial neural network C program as python
extension module.

See:
    http://docs.python.org/2/extending/building.html

"""

from distutils.core import setup, Extension


mod = Extension(
    "khann",
    sources=["src/khann.c", "src/khannmodule.c", "src/hashtable.c"],
    extra_compile_args=["--std=c99"],
    libraries=["mongoc"],
    )

setup(
    name="khann",
    version="1.0",
    description="Kollaborative hosted artificial neural network",
    ext_modules=[mod])

