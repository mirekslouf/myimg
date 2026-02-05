'''
Subpackage: myimg.apps.MDistr
-----------------------------

Calculation of distributions/histograms from various input data.

* The calculation of a simple distribution is straightforward in Python.
* It gets complicated if we have multiple datafiles, various mags/weights ...
* MDistr simplifies the calculation/plotting of various distributions types. 
'''

import importlib

__all__ = ['read','calc','plot']

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
