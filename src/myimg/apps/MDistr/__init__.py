'''
Subpackage: myimg.apps.MDistr
-----------------------------

Calculation of distributions/histograms from various input data.

* The calculation of a simple distribution is straightforward in Python.
* It gets complicated if we have multiple datafiles, various mags/weights ...
* MDistr simplifies the calculation/plotting of various distributions types. 
'''

from myimg.apps.MDistr.data import (
    FileInfo,
    PandasReadOptions,
    CombinedData,
    Statistics,
    Histogram)

from myimg.apps.MDistr.plots import (
    Plots,
    Templates)

__all__ = [
    "FileInfo",
    "PandasReadOptions",
    "CombinedData",
    "Statistics",
    "Histogram",
    "Plots",
    "Templates"]
