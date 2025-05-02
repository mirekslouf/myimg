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
    '''
    Class defining Peaks objects.
    
    * Peaks object = source image + list-of-its-peaks.
    * See __init__ for more information about initial object parameters.
    * More help: https://mirekslouf.github.io/myimg/docs/pdoc.html/myimg.html
    '''

    
    def __init__(self, df=None, img=None, img_name="", file_name="output", messages=False):
        '''
        Initialize Peaks object.

        Parameters
        ----------
        df : pandas.DataFrame object
            DataFrame containing peak coordinates and types.
        img : PIL Image object, optional
            The image associated with the peaks.
        img_name : str, optional
            Name of the image file.

        Returns
        -------
        Peaks object
        '''

        if isinstance(df, pd.DataFrame):
            self.df = df
        elif df is None:
            self.df = pd.DataFrame()
        else:
            print('Error when initializing {myimg.objects.peaks} object.')
            print('The data variable was not in pandas.DataFrame format.')
            print('WARNING: Empty dataframe created instead.')
            sys.exit()

        # Initialize the image and image name
        self.img = img
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
    
    
    def show_in_image(self):
        '''
        Display the image with the peak data overlay (if image and data exist),
        with different colors for different particle types.
    
        Parameters
        ----------
        self : object
            The instance of the class containing the image and peak data.
    
        Returns
        -------
        None
            The method does not return a value but displays an image with
                overlayed peak data.
    
        Raises
        ------
        ValueError
            If the DataFrame does not contain the required columns.
        '''
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
        plt.imshow(self.img)
    
        # Loop through each unique particle type,
        # and plot the peaks with the corresponding color
        for particle_type in self.df['Class'].unique():
            particle_data = self.df[self.df['Class'] == particle_type]
            plt.scatter(particle_data['X'], particle_data['Y'], 
                        c=color_map.get(str(particle_type), 'black'),
                        # Default to black if type is not in the map
                        label=particle_type, 
                        s=25, marker='+')
        
        plt.legend(title="Particle Type", loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.axis("off")
        plt.title(f"Peaks on {self.img_name}")
        plt.show()


   
    def find(self, method='manual', ref=True, mask_path=None, midx=0, thr=0.5, show=True):
        '''
        Create an interactive plot for particle classification.
    
        Parameters
        ----------
        self : object
            The instance of the class that contains the image
            and associated methods.
        method : str, optional
            The method to use for finding peaks.
            Options: 'manual' or 'ccorr'.
            Default is 'manual'.
        ref : bool, optional
            Whether the nanoparticles' coordinates should be refined.
            Default is True.
        mask : array-like or str, optional
            Either a NumPy array representing the mask or a string path
            to a .pkl file containing the mask. Required if method='ccorr'.
    
        Returns
        -------
        None
            This method does not return a value
            but displays an interactive plot for particle classification.
    
        Raises
        ------
        ValueError
            If the specified method is not recognized.
        FileNotFoundError
            If the provided mask path does not exist.
        '''
    
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
                    raise FileNotFoundError(f"Input file not found: {file_path}")
                with open(file_path, 'rb') as f:
                    self.masks[i] = pickle.load(f)
            
            
            # Proceed with detector-based correlation using the mask
            if midx=="all":
                # Average all class-specific masks to get a single "ultimate" mask
                mmask = np.mean(np.stack(list(self.masks.values())), axis=0)
                self.detected = milab.detectors.detector_correlation(self.img, 
                                                         mmask, 
                                                         thr, 
                                                         show)
            else: 
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
        Characterizes image peaks by extracting ROIs, computing features, and 
        selecting the most informative ones for classification.
        
        Parameters
        ----------
        img_path : str
            Path to the image file (e.g., '.tif') to analyze.
        
        peak_path : str
            Path to the pickled file containing detected peak coordinates 
            (typically a DataFrame or array).
        
        mask_path : str
            Directory containing class masks as 'mask1.pkl' to 'mask4.pkl'.
        
        imID : str, optional
            Identifier for the image, used for internal labeling
            Default is 'im0x'
        
        preprocess : bool, optional
            Whether to preprocess the input image (CLAHE, gamma correction, 
            normalization). Default is True.
        
        show : bool, optional
            If True, displays example ROIs and feature visualizations.
            Dfault is False.
        
        Returns
        -------
        None
            This method modifies the internal state of the object by setting 
            attributes such as self.features, self.selection, self.X_train, 
            self.X_test, etc.
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
        self.arr, self.df, _  = milab.roi.prep_data(img_path, peak_path, 
                                             min_xy=20, imID=imID, show=show)
        
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
        
        
        # Show examples of extracted roi if show=True
        if show:
            milab.roi.show_random_rois(self.rois, self.dfs, n=5)
            
        
        # (5) Calculate features
        self.features, _ = milab.features.get_features(self.rois, self.dfs, 
                                                       self.masks,
                                                       show=False)
        
        self.features = self.features.dropna()
        
        if show:
            milab.features.visualize_features(self.features, method="box")
        milab.features.visualize_features(self.features, method="heat")
        
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
        Refines or corrects peak coordinates based on a specified method.
        
        Parameters
        ----------
        image : np.ndarray
            The input image in which the coordinates will be corrected
        
        coords : pandas.DataFrame
            DataFrame containing peak coordinates (e.g., 'X' and 'Y' columns).
        
        s : int, optional
            Half-size of the square ROI to extract around each coordinate. 
            Necessary for "intensity" method. Default is 20.
        
        method : str, optional
            Correction method to use. Currently only 'intensity' is supported 
            (uses intensity-based ROI extraction). Default is 'intensity'.
        
        Returns
        -------
        pandas.DataFrame
            Updated coordinates DataFrame after correction.
        """

        if method == "intensity":
           _, peaks =  milab.roi.get_ROIs(im=image, df=coords, 
                                      s=s, norm=False, show=False)
        else: 
            raise ValueError("Currently, only the 'intensity'\
                             method is supported.")
           
        return peaks
    
    
    def classify(self, data, method='gauss_fit', target=None, estimator=None, param_dist=None, sfeatures=None):
        """
        Classifies input data using the specified classification method.
        
        Parameters
        ----------
        data : pandas.DataFrame or np.ndarray
            The feature data to classify.
        
        method : str, optional
            The classification method to use. Currently, only 'rfc' (Random 
            Forest Classifier) is supported. 
            Default is 'gauss_fit' (placeholder for future extensions).
        
        target : array-like, optional
            Ground truth labels for the data (used for evaluation if provided).
        
        estimator : RandomForestClassifier or str, optional
            A pre-trained classifier or a path to a saved estimator file (.pkl).
            If None, a new RandomForestClassifier will be optimized and trained.
        
        param_dist : dict, optional
            Hyperparameter search space for optimizing the RandomForestClassifier.
            Used only when training a new estimator.
        
        sfeatures : list of str, optional
            List of selected feature names to use for classification. If None, 
            uses `self.selection`.
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels for the input data.
        
        Raises
        ------
        ValueError
            If the provided estimator is neither a valid path nor a classifier 
            instance.
            
        """
        # TODO: calculate only selected features for the testing data   


        if method == "rfc":
            # If no estimator is provided, optimize and train a new Random Forest
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