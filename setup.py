from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'A Python package with functions for WANDA'
LONG_DESCRIPTION = 'My libary of Python functions for the Water Hammer software WANDA'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="wandalib", 
        version=VERSION,
        author="Juan David Guerrero",
        author_email="<juanguerrero09mc@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["pywanda==4.7.0a2", "numpy", "matplotlib", "pandas"], # add any additional packages that 

        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Intended Audience :: Research",
            "Intended Audience :: Industry",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)