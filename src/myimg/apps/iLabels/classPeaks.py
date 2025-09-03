# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:16:17 2025

@author: p-sik
"""

import sys 
import pickle
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
import numpy as np
import myimg.apps.iLabels as milab

class Peaks:
    """
    Container for peak annotations tied to a single image.

    A `Peaks` instance stores:
      - the source image (as a NumPy array or PIL image), and
      - a table of detected/annotated peak positions (and optional labels).

    The peak table is expected to be a `pandas.DataFrame` with at least
    coordinate columns (see `__init__` for details). Use this class as the
    central object for manual/automatic labeling, visualization, and
    feature extraction workflows.

    Notes
    -----
    - See the constructor for required/optional columns in ``df``.
    - For usage examples and API docs, see:
      https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html
    """

    def __init__(self, 
                 df=None, 
                 img=None, 
                 img_name="", 
                 file_name="output", 
                 messages=False):
        """
        Initialize a `Peaks` object.

        Parameters
        ----------
        df : pandas.DataFrame or None, optional
            DataFrame containing peak information. If provided, it should
            include at least coordinate columns (commonly ``'x'`` and ``'y'``,
            float or int). Optional columns such as ``'label'``/``'class'``,
            confidence scores, etc., are supported. If ``None``, an empty
            DataFrame is created.
            
        img : numpy.ndarray or PIL.Image.Image or None, optional
            The image associated with the peaks. If a PIL image is passed, it
            is converted to a NumPy array. If ``None``, image-dependent
            operations will be unavailable until an image is set.
            
        img_name : str, optional
            Name shown in logs/plots to identify the image (e.g., original
            filename). Default is an empty string.
            
        file_name : str, optional
            Base filename used when exporting figures/tables. 
            Default ``"output"``.
            
        messages : bool, optional
            If ``True``, enable verbose/diagnostic prints. Default ``False``.

        Returns
        -------
        None
            Constructor initializes attributes in-place.

        """

        if isinstance(df, pd.DataFrame):
            self.df = df
        elif df is None:
            self.df = pd.DataFrame()
        else:
            print('Error when initializing {myimg.objects.peaks} object.')
            print('The data variable was not in pandas.DataFrame format.')
            print('WARNING: Empty dataframe created instead.')
            sys.exit()

        # --- FIX: ensure image is always numpy array ---
        if img is not None:
            try:
                # If it's a PIL image, convert to numpy
                if hasattr(img, "size") and not hasattr(img, "shape"):
                    self.img = np.array(img)
                else:
                    self.img = img
            except Exception as e:
                print(f"Warning: could not convert image to array ({e})")
                self.img = img
        else:
            self.img = None
    
        self.img_name = img_name
        self.file_name = file_name
        self.messages = messages
            
    
    
    def read(self, filename):
        '''
        Load the peak data from a .pkl file.
    
        Parameters
        ----------
        filename : str
            The path to the .pkl file containing the peak data.
    
        Returns
        -------
        None
        '''
        try:
            self.df = pd.read_pickle(filename)
            # Load the DataFrame from the specified .pkl file
            if self.messages:
                print(f"Data loaded successfully from {filename}")
            # Print success message
        except FileNotFoundError:
            print(f"File {filename} not found.")
            # Print error if file is not found
        except Exception as e:
            print(f"An error occurred: {e}") 
            # Print any other exceptions that occur

    
    def show_as_text(self):
        '''
        Display the peak data as text.
    
        Returns
        -------
        None
        '''
        if self.df is not None:
            print(self.df.to_string(index=False)) 
            # Print the DataFrame as a string without the index
        else:
            print("No data to display. Please read data from a file first.")
            # Print message if no data is available
    
    
    def show_in_image(self, cmap="viridis"):
        """
        Overlay peak locations on the image, colored by class.
    
        Plots ``self.img`` and scatters the peak coordinates from ``self.df``.
        Points are colored by the value in the ``'Class'`` column.
    
        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap used when displaying single-channel (grayscale)
            images. Ignored for RGB/RGBA images. Default is ``"viridis"``.
    
        """
        if self.img is None:
            print("No image to display.")
            return
        if self.df.empty:
            print("No peak data to overlay on the image.")
            return
    
        # Check if the DataFrame contains the required columns
        if 'X' not in self.df.columns or 'Y' not in self.df.columns:
            print("Peak data does not contain 'X' and 'Y' columns.")
            return
        if 'Class' not in self.df.columns:
            print("Peak data does not contain 'Class' column.")
            return
    
        # Define a dictionary mapping particle types to colors
        color_map = {
            '1': 'red',
            '2': 'blue',
            '3': 'green',
            '4': 'purple',
        }
    
        # Plot the image
        plt.imshow(self.img, cmap=cmap)
    
        # Loop through each unique particle type,
        # and plot the peaks with the corresponding color
        for particle_type in self.df['Class'].unique():
            particle_data = self.df[self.df['Class'] == particle_type]
            plt.scatter(particle_data['X'], particle_data['Y'], 
                        c=color_map.get(str(particle_type), 'black'),
                        # Default to black if type is not in the map
                        label=particle_type, 
                        s=25, marker='+')
        
        plt.legend(title="Particle Type", 
                   loc='center left', 
                   bbox_to_anchor=(1.05, 0.5))
        plt.axis("off")
        plt.show()


    def find(self, 
             method='manual', 
             ref=True, 
             mask_path=None, 
             midx=0, 
             thr=0.5, 
             show=True, 
             **kwargs):
        """
        Find peaks either interactively (“manual”) or by norm. cross-correlation
        (“ccorr”) against a reference mask.
    
        Parameters
        ----------
        method : {"manual", "ccorr"}, optional
            Detection mode. "manual" opens an interactive picker; "ccorr" 
            performs normalized cross-correlation with a reference mask. 
            Default "manual".
            
        ref : bool, optional
            Whether to refine detected coordinates after the initial pick 
            (reserved for future use). Default True.
            
        mask_path : str, optional
            Directory containing reference masks saved as ``mask1.pkl``,
            ``mask2.pkl``,... Required when ``method="ccorr"``.
            
        midx : int, optional
            Mask index to use. Default 0.
            
        thr : float, optional
            NCC threshold in [0, 1]. Detections with correlation below this are
            discarded. Default 0.5.
            
        show : bool, optional
            Visualize the detections/overlay. Default True.
            
        **kwargs
            Reserved for passing additional options to the interactive plotter
            or detector implementation (e.g., debug flags).
    
        Returns
        -------
        detected : Any or None
            For ``method="ccorr"``, returns whatever 
            ``milab.detectors.detector_correlation`` returns (e.g., a DataFrame 
            of coordinates/scores). For ``method="manual"``, returns ``None`` 
            after displaying the picker.

        """
    
        if method == "manual":
            from myimg.utils.iplot import interactive_plot, default_plot_params
    
            # Generate and display the interactive plot for manual annotation
            fig, ax = interactive_plot(self.img, 
                                       default_plot_params(self.img),
                                       filename=self.file_name, 
                                       messages=self.messages)
            plt.show()
            
    
        elif method == "ccorr":
            # Load masks
            self.masks = {}
            for i in range(1, 5):
                file_path = os.path.join(mask_path, f"mask{i}.pkl")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Input file not found: {file_path}")
                with open(file_path, 'rb') as f:
                    self.masks[i] = pickle.load(f)

                    
            # Proceed with detector-based correlation using the mask
            self.detected = milab.detectors.detector_correlation(self.img, 
                                                     self.masks[midx], 
                                                     thr, 
                                                     show)
            
            return self.detected
    
        else:
            raise ValueError("Invalid detection method. Use 'manual'/'ccorr'.")
        
        
        # if ref:
        #     # TODO: apply correct method
        #     pass
            
    

    def characterize(self, img_path, peak_path, mask_path, 
                     imID='im0x', preprocess=True, show=False):
        """
        Extract ROIs around peaks, compute features, and select the most 
        informative subset for downstream classification.
    
        Pipeline
        --------
        1) Load class masks from ``mask_path`` (``mask1.pkl``..``mask4.pkl``).
        2) (Optional) Preprocess the image (CLAHE, gamma, normalization).
        3) Load peaks and prepare data for ROI extraction.
        4) Extract ROIs (square windows) centered on peaks; ROI size is derived
           from the mask size.
        5) Compute features for each ROI (intensity, morphology, correlation,
           Gaussian fit, etc.).
        6) Split into train/test and perform forward feature selection (SFS).
    
        Parameters
        ----------
        img_path : str
            Path to the source image (e.g., ``.tif``).
            
        peak_path : str
            Path to a pickled peaks file (e.g., a DataFrame with at least 
            columns ``'X'``, ``'Y'``, and optionally class labels), consumed by
            ``milab.roi.prep_data``.
            
        mask_path : str
            Directory containing class masks saved as ``maskX.pkl``.
            The first mask's size is used to derive the ROI half-size.
            
        imID : str, optional
            Image identifier used for labeling/traceability of ROIs. 
            Default "im0x".
            
        preprocess : bool, optional
            If True, apply CLAHE, gamma correction, and normalization before 
            ROI extraction. Default True.
            
        show : bool, optional
            If True, display example ROIs and/or intermediate visualizations.
            Default False.
    
        Returns
        -------
        None
            This method updates the object in-place with:
            - ``self.masks``       : dict of loaded masks
            - ``self.pimg``        : preprocessed image (if enabled)
            - ``self.arr, self.df``: prepared data/peaks table
            - ``self.rois``        : list of (roi_array, imID)
            - ``self.features``    : feature table (NaNs dropped)
            - ``self.selection``   : feature selector / chosen feature set
            - ``self.X_train, self.X_test, self.y_train, self.y_test``
        """
        # (1) Load masks
        self.masks = {}
        for i in range(1, 5):
            file_path = os.path.join(mask_path, f"mask{i}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
            with open(file_path, 'rb') as f:
                self.masks[i] = pickle.load(f)
        
        # Calculate ROI shape based on the mask shape
        s = int(self.masks[i].shape[0]/2)
        
        # (2) Preprocess image
        if preprocess:
           self.pimg =  milab.roi.preprocess_image(self.img, 
                                               apply_clahe=True, 
                                               gamma=1.2, 
                                               normalize=True)
        
        # (3) Prepare data for ROI extraction
        self.arr, self.df, peaks = milab.roi.prep_data(
            img_path, peak_path, min_xy=20, imID=imID, show=False)
        
        # keep a reference to the Peaks object
        self.peaks = peaks
        
        # (4) Extract ROIs from image
        self.rois, self.dfs = [], []
        rois, df = milab.roi.get_ROIs(im=self.pimg, 
                                            df=self.df, 
                                            s=s, 
                                            norm=False, 
                                            show=show)
        

        self.dfs.append(df)        
        self.rois = [(roi, imID) for roi in rois]        
        self.dfs = pd.concat(self.dfs, ignore_index=True)
        
        # (5) Calculate features
        self.features, _ = milab.features.get_features(self.rois, self.dfs, 
                                                       self.masks,
                                                       show=False)

        self.features = self.features.dropna()
                
        # (6) Select features
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = \
            milab.classifiers.dataset(self.features)
        
        # Select features
        self.selection = milab.classifiers.select_features(self.X_train, 
                                                           self.y_train, 
                                                           num=5, 
                                                           estimator=None)
        
        return 
    
    
    def correct(image, coords, s=20, method='intensity'):
        """
        Refine peak coordinates by re-centering each peak to the local
        intensity maximum within a square ROI.
    
        Parameters
        ----------
        image : numpy.ndarray
            2D grayscale image (H×W) in which peaks were detected.
            
        coords : pandas.DataFrame
            Table of peak locations with at least columns ``'X'`` and ``'Y'``.
            (By convention, ``X`` is the column/index along width, ``Y`` is the
            row/index along height.)
            
        s : int, optional
            Half-size of the square ROI (in pixels) extracted around each
            (X, Y) location, i.e. the ROI side is ``2*s+1``. Default ``20``.
            
        method : {'intensity'}, optional
            Refinement strategy. Currently only ``'intensity'`` is supported,
            which recenters coordinates to the maximum-intensity pixel within
            the ROI. Default ``'intensity'``.
    
        Returns
        -------
        pandas.DataFrame
            A new DataFrame with corrected peak coordinates (and any 
            accompanying metadata preserved by the underlying ROI routine).
    
        Raises
        ------
        ValueError
            If an unsupported ``method`` is requested.
    
        Notes
        -----
        This function delegates ROI extraction and coordinate refinement to
        ``milab.roi.get_ROIs`` and returns its updated coordinates table.
        Make sure your coordinates are within image bounds; peaks too close
        to the border may yield clipped ROIs depending on ``get_ROIs`` behavior
        """

        if method == "intensity":
           _, peaks =  milab.roi.get_ROIs(im=image, df=coords, 
                                      s=s, norm=False, show=False)
        else: 
            raise ValueError("Currently, only the 'intensity'\
                             method is supported.")
           
        return peaks
    
    
    def classify(self, 
                 data, 
                 method='gauss_fit', 
                 target=None, 
                 estimator=None, 
                 param_dist=None, 
                 sfeatures=None):
        """
        Classify samples using the selected method.
    
        Currently implemented method(s)
        --------------------------------
        'rfc' : Random Forest Classifier
            If no estimator is provided, an RFC is optimized via
            `milab.classifiers.get_optimal_rfc` using `self.X_train` or
            `self.y_train` and (optionally) `param_dist`. The optimized model 
            is then fitted and used to predict on `data`. 
            If `estimator` is provided (either a fitted classifier instance or 
            a path to a .pkl file), it is used directly.
    
        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Feature matrix to classify. If feature selection is used, 
            the columns should at least include `sfeatures` 
            (or `self.selection` if omitted).
            
        method : str, optional
            Classification backend. Only 'rfc' is supported at present.
            (The default 'gauss_fit' is a placeholder and not implemented.)
            
        target : array-like, optional
            Ground-truth labels (used for evaluation inside
            `milab.classifiers.predicting` if supplied).
            
        estimator : sklearn.base.ClassifierMixin or str, optional
            Either a fitted scikit-learn classifier instance or a filesystem 
            path to a serialized estimator (.pkl) loadable via `joblib.load`. 
            If None, a new RFC is optimized and trained.
            
        param_dist : dict, optional
            Hyperparameter search space for optimizing the RFC.
            Only used when `estimator` is None.
            
        sfeatures : list of str, optional
            Explicit list of feature names to use for training/prediction. 
            If None, falls back to `self.selection`.
    
        Returns
        -------
        y_pred : numpy.ndarray
            Predicted class labels for `data`.
    
        """
        if method == "rfc":
            # If no estimator is provided, optimize and train a new RFC
            if estimator is None:
                # Perform hyperparameter search and get an optimized classifier

                rfc_opt = milab.classifiers.get_optimal_rfc(self.X_train,
                                                  self.y_train, 
                                                  param_dist)
                
                # Fit the optimized classifier to the training data
                self.rfc_fit, _ = milab.classifiers.fitting(self.X_train, 
                                                  self.y_train, 
                                                  estimator=rfc_opt, 
                                                  reports=True, 
                                                  sfeatures=self.selection)
                
                # Predict class labels on the input data using the classifier
                self.y_pred = milab.classifiers.predicting(data, 
                                                 estimator=self.rfc_fit, 
                                                 sfeatures=self.selection, 
                                                 y_test=target)
            else: 
                # Load the pre-trained classifier if estimator is a file path
                if isinstance(estimator, str):
                    estimator = joblib.load(estimator)

                elif not isinstance(estimator, ClassifierMixin):
                    # Raise error if estimator is not a valid classifier object
                    raise ValueError("Provided estimator is invalid.")
                
                # Use default selected features if none are provided
                if sfeatures is None:
                    sfeatures = self.selection
                
                # Predict class labels using the provided classifier
                self.y_pred = milab.classifiers.predicting(data, 
                                                 estimator, 
                                                 sfeatures, 
                                                 y_test=target)
                
        return self.y_pred