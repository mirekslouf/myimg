# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:50:17 2025

@author: p-sik
"""

# src/myimg/apps/iLabels/__init__.py
import importlib

__all__ = ["features", "roi", "classPeaks", "classifiers", "detectors"]

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
