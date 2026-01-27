# -*- coding: utf-8 -*-
"""
iLabels: interactive labeling, ROI extraction, and classification tools.

The :mod:`myimg.apps.iLabels` package provides a modular toolkit for
interactive and automated annotation of particle-like objects in microscopy
images (e.g. SEM/TEM). It is designed to support the full workflow from
manual labeling through feature extraction to supervised classification.

Submodules
----------
features
    Feature extraction utilities operating on ROIs, including:
    - intensity statistics,
    - morphological descriptors,
    - normalized cross-correlation features,
    - 2D Gaussian fit parameters and derived quantities.

roi
    Image preprocessing and Region of Interest (ROI) utilities:
    - image loading and normalization,
    - ROI extraction and re-centering,
    - generation of class-averaged masks for template matching.

classPeaks
    Defines the :class:`~myimg.apps.iLabels.classPeaks.Peaks` class, which
    binds an image to a peak table and provides high-level methods for:
    - visualization,
    - interactive/manual annotation,
    - detector-based peak finding,
    - ROI characterization and feature selection,
    - downstream classification.

classifiers
    Machine-learning helpers for supervised classification, including:
    - dataset preparation and train/test splitting,
    - feature selection,
    - Random Forest training, optimization, and prediction.

detectors
    Peak detection algorithms, including:
    - normalized cross-correlation (NCC) template matching,
    - correlation-based detectors using reference masks,
    - post-processing steps such as duplicate removal and spatial filtering.


Typical usage
-------------
>>> import joblib, pickle
>>> import pandas as pd
>>> import myimg.apps.iLabels.classifiers as miclass
>>>
>>> # Load a suitable classification model
>>> classifier = joblib.load("rfc_model.joblib")
>>>
>>> # Load the list of previously selected features
>>> with open("selected_features.pkl", "rb") as f:
>>>    selection = pickle.load(f)
>>> 
>>> # Load dataset represented by features 
>>> features = pd.read_pickle("features.pkl")
>>>
>>> # Classification
>>> y_pred = miclass.predicting(features, 
>>>                             estimator=classifier, 
>>>                             sfeatures=selection)

"""


# src/myimg/apps/iLabels/__init__.py
import importlib

__all__ = ["features", "roi", "classPeaks", "classifiers", "detectors"]

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
