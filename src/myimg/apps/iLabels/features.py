# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:08:54 2025

@author: p-sik
"""
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk
from scipy.optimize import curve_fit
from skimage.feature import match_template
import matplotlib.pyplot as plt


def get_features(rois, df, masks, show=False):
    """
    Extracts and combines multiple types of features from ROI images:
        - statistical
        - morphological
        - correlation-based
        - Gaussian-fitting features.

    Parameters:
    -----------
    rois : list of tuple
        List of tuples where each tuple is (roi, label), with:
        - roi : np.ndarray
            A 2D array representing the image of the ROI.
        - label : str or int
            Identifier or class label for the ROI.

    df : pandas.DataFrame
        DataFrame containing metadata for each ROI. It is expected to contain
        information like coordinates, class labels, and notes.

    masks : list of np.ndarray
        List of binary masks (same size as ROIs), where each mask defines the 
        region of interest used for correlation feature extraction.

    show : bool, optional 
        If True, displays visualizations during feature extraction (e.g., for 
        verification). Default is False.

    Returns:
    --------
    combined_props : pandas.DataFrame
        A DataFrame containing all extracted features:
        - Statistical and morphological features
        - Gaussian fit parameters
        - Correlation-based features
        The class column is retained only once to avoid duplication.

    propsArr : np.ndarray
        A NumPy array containing additional features or summary statistics, 
        typically used for low-dimensional numerical representation.

    Notes:
    ------
    - Internally calls:
        - `extended_features()` for statistical and morphological features
        - `corr_features()` for correlation metrics using masks
        - `gauss_params()` and `gauss_features()` for Gaussian fitting

    """
    
    # Access the first element (the ROI image) of each tuple
    rois_only = [roi_data[0] for roi_data in rois]  
    
    # Compute statistical and morphological features 
    propsFloat, propsArr = extended_features(rois_only, df, show)
    
    # Compute correlation features
    propsCorr = corr_features(rois_only, df, masks)
    
    # Compute gaussian features
    gauss = gauss_params(rois_only, df)
    propsGauss = gauss_features(gauss)

    # Remove 'class' column from one of the dataframes (propsGauss or propsFloat)
    propsGauss = propsGauss.drop(columns=['class'])
    
    # Concatenate the two dataframes along axis 1 (columns)
    combined_props = pd.concat([propsFloat, propsGauss, propsCorr], axis=1)

    return combined_props, propsArr


def extended_features(rois, df, show=True):
    """
    Extracts and extends feature sets from a list of Region of Interest (ROI)
    images and returns two DataFrames containing scalar and array-based features 
    respectively.

    The function computes basic ROI features via `roi_features()` and then 
    performs additional morphological and shape analysis on each ROI, including 
    thresholding (Otsu), morphological filtering, region labeling, and extraction 
    of geometric and moment-based properties. 

    Parameters
    ----------
    rois : list of ndarray
        List of 2D NumPy arrays, each representing an ROI (grayscale image).
    
    df : pandas.DataFrame
        DataFrame containing metadata associated with each ROI, including a 
        'Class' column that specifies the particle category or label.
    
    show : bool, optional (default=True)
        If True, displays boxplots of each numeric feature grouped by class. 
        Skips features that are non-numeric or contain NaN/inf.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame with original `df` columns, features from `roi_features`, 
        and additional scalar morphological features such as area, eccentricity, 
        solidity, etc.
    
    df2 : pandas.DataFrame
        DataFrame with original `df` columns and array-based features including 
        image moments and central moments for each ROI.
    
    Notes
    -----
    - For ROIs that are empty or contain uniform values, NaNs are assigned to 
      all extracted features.
    - Features extracted include:
        - Scalar: area, convex area, eccentricity, solidity, equivalent diameter, 
                  extent, number of pixels, orientation, and perimeter.
        - Array: raw spatial moments and central moments (both as 3x3 arrays).
    - The function assumes that the `df` DataFrame has the same number of rows 
      as the number of ROIs provided, and that it includes a 'Class' column for 
      grouping.
    
    Raises
    ------
    Handles all exceptions during individual ROI processing and prints an error 
    message.
    """

    binary = []
    labeled = []
    
    # This will hold dictionaries for each ROI with selected properties
    props_float = []  
    props_arr = []  

    roi_df = roi_features(rois, df, show=False, out=0)
    
    for i, roi in enumerate(rois):        # Skip or handle empty arrays
        if roi.size == 0:
            binary.append(None)
            labeled.append(None)
            roi_properties1 = {'area': np.nan, 
                                'area_convex' : np.nan,
                                'eccentricity': np.nan, 
                                'solidity': np.nan,
                                'equivalent_diameter_area' : np.nan,
                                'extent' : np.nan,
                                'num_pixels' : np.nan,
                                'orientation' : np.nan,
                                'perimeter' : np.nan,
                                'otsu' : np.nan,
                                'class': df.Class.iloc[i].item() \
                                    if hasattr(df.Class.iloc[i], 'item') 
                                    else df.Class.iloc[i],
                                }
            
            roi_properties2 = {'moments': np.nan,
                              'moments_central' : np.nan,
                              'class': [i.item() if hasattr(i, 'item') 
                                        else i for i in df.Class],
                              }
            props_float.append(roi_properties1)
            props_arr.append(roi_properties2)
            continue

        # Replace NaNs with 0 (or another strategy)
        roi = np.nan_to_num(roi)
        
        # Skip if all values are the same (e.g., all 0)
        if np.all(roi == roi.flat[0]):
            binary.append(None)
            labeled.append(None)
            roi_properties1 = {'area': np.nan, 
                                'area_convex' : np.nan,
                                'eccentricity': np.nan, 
                                'solidity': np.nan,
                                'equivalent_diameter_area' : np.nan,
                                'extent' : np.nan,
                                'num_pixels' : np.nan,
                                'orientation' : np.nan,
                                'perimeter' : np.nan,
                                'otsu' : np.nan,
                                'class': df.Class.iloc[i].item() \
                                    if hasattr(df.Class.iloc[i], 'item') 
                                    else df.Class.iloc[i],
                                }
            
            roi_properties2 = {'moments': np.nan,
                              'moments_central' : np.nan,
                              'class': df.Class.iloc[i].item() \
                                  if hasattr(df.Class.iloc[i], 'item') 
                                  else df.Class.iloc[i],
                              }
                
            props_float.append(roi_properties1)
            props_arr.append(roi_properties2)
            continue

        try:
            # Get threshold using Otsu method
            # TODO zamyslet se, jestli by nebylo lepsi pouzivat 1 globalni 
            # threshold pro vsechny nanocastice
            thresh = threshold_otsu(roi)
            binim = roi > thresh
            
            # Apply morphological opening to remove small noise
            binim_filtered = opening(binim, disk(1))  
            binary.append(binim_filtered)
            
            # Label the filtered binary image
            lab = label(binim_filtered)
            labeled.append(lab) 
            
            # Calculate region properties
            regions = regionprops(lab)
            
            # Initialize a dictionary to store selected properties for this ROI
            roi_properties1 = {'area': np.nan, 
                                'area_convex' : np.nan,
                                'eccentricity': np.nan, 
                                'solidity': np.nan,
                                'equivalent_diameter_area' : np.nan,
                                'extent' : np.nan,
                                'num_pixels' : np.nan,
                                'orientation' : np.nan,
                                'perimeter' : np.nan,
                                'otsu' : np.nan,
                                'class': df.Class.iloc[i].item() \
                                    if hasattr(df.Class.iloc[i], 'item') 
                                    else df.Class.iloc[i],
                                }
            
            roi_properties2 = {'moments': np.nan,
                              'moments_central' : np.nan,
                              'class': df.Class.iloc[i].item() \
                                  if hasattr(df.Class.iloc[i], 'item') 
                                  else df.Class.iloc[i],
                              }
            
            if regions:
                region = regions[0]
                
                # Add properties you want to track
                # (1) AREA OF THE REGION (number of pixels of the region 
                # scaled by pixel-area)
                roi_properties1['area'] = region.area 
                
                # (2) AREA OF THE CONVEX HULL IMAGE (the smallest convex polygon 
                # that encloses the region)
                roi_properties1['area_convex'] = region.area_convex
                
                # (3) ECCENTRICITY (the ratio of the focal distance (distance 
                # between focal points) over the major axis length. The value 
                # is in the interval [0, 1). When it is 0, the ellipse becomes 
                # a circle.
                roi_properties1['eccentricity'] = region.eccentricity
                
                # (4) SOLIDITY (Ratio of pixels in the region to pixels of the 
                # convex hull image.)
                roi_properties1['solidity'] = region.solidity
                
                # (5) EQUIVALENT DIAMETER AREA (the diameter of a circle with 
                # the same area as the region)
                roi_properties1['equivalent_diameter_area'] = \
                    region.equivalent_diameter_area
                
                # (6) EXTENT (ratio of pixels in the region to pixels in the 
                # total bounding box. Computed as area / (rows * cols))
                roi_properties1['extent'] = region.extent
                
                # (7) NUMBER OF PIXELS (number of foreground pixels)
                roi_properties1['num_pixels'] = region.num_pixels
                
                # (8) ORIENTATION (angle between the 0th axis (rows) and the 
                # major axis of the ellipse that has the same second moments 
                # as the region, ranging from -pi/2 to pi/2 counter-clockwise.)
                roi_properties1['orientation'] = region.orientation
                
                # (9) PERIMETER (Perimeter of object which approximates the 
                # contour as a line through the centers of border pixels using 
                # a 4-connectivity.)
                roi_properties1['perimeter'] = region.perimeter
                
                # (10) OTSU THRESHOLD
                roi_properties1['otsu'] = thresh
                
                # (10) MOMENTS (spatial moments up to 3rd order, 3x3 array)
                roi_properties2['moments'] = region.moments
                
                # (11) MOMENTS CENTRAL (central moments (translation invariant) 
                # up to 3rd order, 3x3 array)
                roi_properties2['moments_central'] = region.moments_central
                
            
                

            # Append the properties dictionary for this ROI
            # FLOAT properties
            props_float.append(roi_properties1)
            
            # ARRAY properties
            props_arr.append(roi_properties2)
            
        except Exception as e:
            # Catch unexpected errors and handle them gracefully
            binary.append(None)
            labeled.append(None)
            
            # FLOAT properties
            props_float.append({'area': np.nan, 
                                    'area_convex' : np.nan,
                                    'eccentricity': np.nan, 
                                    'solidity': np.nan,
                                    'equivalent_diameter_area' : np.nan,
                                    'extent' : np.nan,
                                    'num_pixels' : np.nan,
                                    'orientation' : np.nan,
                                    'perimeter' : np.nan,
                                    'otsu' : np.nan,
                                    'class': df.Class.iloc[i].item() \
                                        if hasattr(df.Class.iloc[i], 'item') 
                                        else df.Class.iloc[i],
                                    })
            
            # ARRAY properties
            props_arr.append({'moments': np.nan,
                              'moments_central' : np.nan,
                              'class': df.Class.iloc[i].item() \
                                    if hasattr(df.Class.iloc[i], 'item') 
                                    else df.Class.iloc[i],
                                })
            
            
            print(f"Skipping ROI due to error: {e}")
    
    # Convert the list of dictionaries to a pandas DataFrame
    properties_df1 = pd.DataFrame(props_float)

    properties_df2 = pd.DataFrame(props_arr)

    # If any rows have missing data (None), fill those rows with NaN
    properties_df1 = properties_df1.fillna(np.nan)  
    properties_df2 = properties_df2.fillna(np.nan)  
                 
    
    # Drop duplicate 'class' column to avoid conflicts
    roi_df = roi_df.drop(columns=['class'])
    properties_df1 = properties_df1.drop(columns=['class'])
    properties_df2 = properties_df2.drop(columns=['class'])


    # Append the features to the original df
    df1 = pd.concat([df.reset_index(drop=True),
                     roi_df.reset_index(drop=True),
                    properties_df1.reset_index(drop=True)], 
                   axis=1)
    df2 = pd.concat([df.reset_index(drop=True), 
                    properties_df2.reset_index(drop=True)], 
                   axis=1)
    
    if show:
        for feat in df1.columns[4:]:
            if np.issubdtype(df1[feat].dtype, np.number) and np.isfinite(df1[feat]).all():
                plt.figure(figsize=(6, 4))
                sns.boxplot(x='Class', y=feat, data=df1, palette='Set2')
                plt.title(f'Distribution of {feat} across classes')
                plt.xlabel('Class')
                plt.ylabel(feat)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Skipping {feat} - non-numeric or contains inf/NaN.")
                
    
    return df1, df2 
        

def roi_features(rois, df, show=True, out=1):
    """
    Extracts intensity-based statistical features from a list of ROI images and 
    optionally visualizes feature distributions across classes.
    
    This function computes various intensity statistics such as max, min, mean,
    median, standard deviation, variance, skewness, and kurtosis for each
    Region of Interest (ROI). It associates these features with the corresponding 
    class labels from the input DataFrame.
    
    Parameters
    ----------
    rois : list of ndarray
        List of 2D NumPy arrays representing grayscale ROI images.
    
    df : pandas.DataFrame
        DataFrame containing metadata for each ROI. Must include a 'Class' 
        column specifying the category of each ROI.
    
    show : bool, optional (default=True)
        If True, generates boxplots of each feature grouped by class for visual 
        comparison.
    
    out : int, optional (default=1)
        Determines the format of the output:
        - If 0: returns a new DataFrame containing only the computed features 
                and class labels.
        - If 1: appends the computed features to the original DataFrame and 
                returns the result.
    
    Returns
    -------
    pandas.DataFrame
        Depending on `out`:
        - If out == 0: returns a new DataFrame with computed features and 
                       'class' column.
        - If out == 1: returns the original DataFrame with added columns for 
                        each computed feature.
    
    Notes
    -----
    - For empty ROIs, NaN is assigned to all computed features.
    - The 'Class' column is extracted from `df.Class` and is expected to be 
      convertible to scalar.
    
    """

    data = {
        # Maximum intensity
        'max'   : [np.max(roi) if roi.size > 0 else np.nan for roi in rois],
        # Minimum intensity
        'min'   : [np.min(roi) if roi.size > 0 else np.nan for roi in rois],
        # Mean intensity
        'mean'  : [np.mean(roi) if roi.size > 0 else np.nan for roi in rois],
        # Median intensity
        'median': [np.median(roi) if roi.size > 0 else np.nan for roi in rois],
        # Standard deviation
        'stdev' : [np.std(roi) if roi.size > 0 else np.nan for roi in rois],
        # Variance
        'var'   : [np.var(roi) if roi.size > 0 else np.nan for roi in rois],
        # Skewness (asymmetry of pixel intensity)
        'skew'  : [scipy.stats.skew(roi.flatten()) \
                   if roi.size > 0 else np.nan for roi in rois],
        # Kurtosis (peakedness of intensity distribution)
        'kurt'  : [scipy.stats.kurtosis(roi.flatten()) \
                   if roi.size > 0 else np.nan for roi in rois],
        # True Class
        'class': [i.item() if hasattr(i, 'item') else i for i in df.Class]
    }
    
    data = pd.DataFrame(data)
    
    if show:
        # Omit the last key (assumed to be 'class')
        for feat in list(data)[:-1]:  
                plt.figure(figsize=(6, 4))
                sns.boxplot(x='class', y=feat, data=data, palette='Set2')
                plt.title(f'Distribution of {feat} across classes')
                plt.xlabel('Class')
                plt.ylabel(feat)
                plt.tight_layout()
                plt.show()
                

    
    if out==0: 
        # return new attributes
        return data
    
    elif out==1: 
        # return all attributes
        # Drop duplicate 'class' column to avoid conflicts
        data = data.drop(columns=['class'])

        # Append the features to the original df
        df = pd.concat([df.reset_index(drop=True), 
                        data.reset_index(drop=True)], 
                       axis=1)
        return df


def corr_features(rois, df, masks):
    """
    Calculates image-based correlation (normalized cross-correlation)
    between each ROI and each class-average mask using template matching.

    Parameters:
    -----------
    rois : list of np.ndarray
        List of 2D ROI images (grayscale). Each ROI should be of the same shape 
        as the corresponding masks for valid correlation computation.

    df : pandas.DataFrame
        DataFrame with one row per ROI. It must have at least the same length 
        as `rois`. Expected to include a 'Class' column, although it's not used 
        directly in this function.

    masks : dict or list of np.ndarray
        Class-average masks for template matching. Can be:
            - A dictionary with keys like 'CorrCL1', 'CorrCL2', ..., 'CorrCL4'.
            - A list of four masks (assumed ordered by class index 1 to 4).

    Returns:
    --------
    pd.DataFrame
        DataFrame of shape (n_rois, 5), with:
            - One column per class: correlation scores 
            - A 'bestMatch' column: integer from 1 to 4, indicating the class 
                                    with max correlation.
    """

    # Convert list of masks to a dict with predefined class labels if needed
    if isinstance(masks, list):
        class_names = ["CorrCL1", "CorrCL2", "CorrCL3", "CorrCL4"]
        mask_dict = {name: m for name, m in zip(class_names, masks)}
    else:
        mask_dict = masks
        class_names = list(mask_dict.keys())

    # Will hold correlation scores per ROI
    results = []  

    # Loop over each ROI image
    for roi in rois:
        row = []
        
        # Loop over each class mask and compute normalized cross-correlation
        for cl in class_names:
            mask = mask_dict[cl]
            
            # Check shape compatibility
            if roi.shape != mask.shape:
                raise ValueError("ROI and mask shapes must match.")

            # Compute normalized cross-correlation using match_template
            # Since the ROI and mask are same-sized, the result is a single value
            ncc_score = match_template(roi, mask, pad_input=False)[0][0]
            row.append(ncc_score)
        
        # Save results
        results.append(row)

    # Create a DataFrame from the correlation scores
    corr_df = pd.DataFrame(results, columns=[f"ccorr{c}" for c in class_names])

    # Identify the highest correlation score and corresponding class index
    corr_df["maxCorr"] = corr_df.max(axis=1)
    
    # Find which column had the max
    corr_df["bestMatch"] = corr_df.iloc[:, :4].idxmax(axis=1)  

    # Extract the class number from the column name, e.g., 'ccorrCorrCL2' → 2
    corr_df["bestMatch"] = corr_df["bestMatch"].str.extract(r'(\d)$').astype(int)

    # Drop intermediate maxCorr column (optional)
    corr_df = corr_df.drop(columns=["maxCorr"])

    return corr_df


def gauss_features(df_params):
    """
    Computes derived features from 2D Gaussian parameters.

    Parameters
    ----------
    df_params : pandas.DataFrame
        A DataFrame with columns: ['class', 'amplitude', 'sigma_x', 'sigma_y', 
         'theta', 'offset']

    Returns
    -------
    df_features : pandas.DataFrame
        The input DataFrame with additional derived features: fwhm_x, fwhm_y, 
        area, eccentricity, ellipticity, orientation_deg, peak_intensity, 
        contrast, integrated_intensity
    """
    # Create a copy of the input dataframe to avoid modifying the original one
    df = df_params.copy()

    # Constants
    # (a) The factor for calculating FWHM from sigma (standard deviation)
    FWHM_factor = 2.355
    
    # (b) Pi constant for area and other calculations
    pi = np.pi

    # Derived features based on Gaussian parameters
    # (1) Full width at half maximum (FWHM) along x and y axes
    df['fwhm_x'] = FWHM_factor * df['sigma_x']
    df['fwhm_y'] = FWHM_factor * df['sigma_y']
    
    # (2) Area of the Gaussian shape (in terms of the standard deviations)
    df['area2'] = 2 * pi * df['sigma_x'] * df['sigma_y']
    
    # (3) Eccentricity (how elongated the shape is, 0 = circular, 1 = linear)
    df['eccentricity2'] = np.sqrt(1-(np.minimum(df['sigma_x'],df['sigma_y'])**2/ 
                                np.maximum(df['sigma_x'], df['sigma_y'])**2))
    
    # (4) Ellipticity (ratio of sigma_x to sigma_y)
    df['ellipticity2'] = df['sigma_x']/df['sigma_y']
    
    # (5) Orientation (the angle of rotation of the Gaussian in radians, 
    #     converted to degrees)
    df['orientation_deg'] = np.degrees(df['theta'])
    
    # (6) Peak intensity (the intensity at the peak of the Gaussian)
    df['peak_intensity'] = df['amplitude'] + df['offset']
    
    # (7) Contrast (normalized contrast of the peak)
    #     A small epsilon is added to avoid division by zero for cases where 
    #     amplitude + offset is too small
    df['contrast2'] = df['amplitude'] / (df['amplitude'] + df['offset'] + 1e-8)  
    
    # (8) Integrated intensity (total intensity under the Gaussian, related to
    #     its area)
    df['integrated_intensity'] = df['amplitude']*2*pi*df['sigma_x']*df['sigma_y']
    
    # Drop the 'x0' and 'y0' columns 
    # errors='ignore' to handle if 'x0' or 'y0' is missing
    df = df.drop(columns=['x0'])
    df = df.drop(columns=['y0'])


    return df


def gauss_params(image_list, df):
    """
    Fits a 2D Gaussian function to each image (ROI) in the given list and 
    returns the estimated parameters for each fit in a DataFrame.

    Parameters
    ----------
    image_list : list of np.ndarray
        A list of 2D NumPy arrays representing grayscale image regions (ROIs) 
        where a Gaussian-like spot is expected.
    df : pandas.DataFrame
        DataFrame containing metadata for each ROI, including a 'Class' column 
        that assigns a label to each image.

    Returns
    -------
    df_params : pandas.DataFrame
        A DataFrame where each row corresponds to the parameters 
        of a 2D Gaussian fit for an image in `image_list`. If a fit fails, 
        the row contains NaNs.

        Columns: class (label of the ROI), amplitude (peak height of the Gauss),
                 x0, y0 (center coordinates), sigma_x (std deviation in x dir.),
                 sigma_y  (std deviation in y direction), theta (rotation angle 
                 of the Gaussian ellipse), offset (constant background level)
    """
    # Initialize variables
    param_names = ['class','amplitude','x0','y0','sigma_x','sigma_y','theta', 
                   'offset']
    results = []

    for image, clsx in zip(image_list, df['Class']):
        try:
            # Coordinate grid
            x = np.linspace(0, image.shape[1] - 1, image.shape[1])
            y = np.linspace(0, image.shape[0] - 1, image.shape[0])
            x, y = np.meshgrid(x, y)
            x_data = np.vstack((x.ravel(), y.ravel()))

            # Initial guess
            amplitude = np.max(image)
            xo = image.shape[1] / 2
            yo = image.shape[0] / 2
            sigma_x = sigma_y = 5
            theta = 0
            offset = np.min(image)

            initial_guess = (amplitude, 
                             xo, yo, 
                             sigma_x, sigma_y, 
                             theta, 
                             offset)

            # Fit
            popt, _ = curve_fit(gauss2D, 
                                x_data, 
                                image.ravel(), 
                                p0=initial_guess, 
                                maxfev=10000)
            
            results.append([clsx] + list(popt))

        except Exception:
            # On failure, append NaNs
            results.append([clsx] + [np.nan] * (len(param_names) - 1))

    df_params = pd.DataFrame(results, columns=param_names)
    return df_params


def gauss2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Computes a 2D Gaussian function with elliptical shape and rotation.
    
    This function is typically used for curve fitting on 2D image data, 
    such as intensity spots or blobs.
    
    Parameters
    ----------
    xy : tuple of np.ndarray
        A tuple of two flattened coordinate arrays (x, y), typically created
        from a meshgrid and then stacked for curve fitting. Shape of each array 
        should be (N,), where N is the number of pixels.
    
    amplitude : float
        Peak height of the Gaussian.
    
    xo : float
        X-coordinate of the Gaussian center.
    
    yo : float
        Y-coordinate of the Gaussian center.
    
    sigma_x : float
        Standard deviation in the x-direction.
    
    sigma_y : float
        Standard deviation in the y-direction.
    
    theta : float
        Rotation angle of the Gaussian (in radians), counterclockwise.
    
    offset : float
        Constant offset or background intensity.
    
    Returns
    -------
    g : np.ndarray
        The 2D Gaussian function values evaluated at the (x, y) coordinates,
        returned as a 1D array (raveled). This is required by fitting functions
        like `scipy.optimize.curve_fit`.
    """
    # Unpack flattened coordinate arrays
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    
    # Pre-compute constants for rotated Gaussian
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    
    # Compute 2D Gaussian function values
    g = offset+amplitude*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
    
   
    return g.ravel()  # Flatten for compatibility with curve_fit


def visualize_features(df, method="box", class_col=None):
    """
    Visualize features in the dataframe using various methods.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing feature columns and a class label column.
    method : str, optional (default="box")
        Visualization method to use. Options are:
            - "box" : Individual box plots for each feature grouped by class.
            - "pair": Pairwise scatter plots between features, color-coded 
                      by class.
            - "heat": Heatmap of feature correlations.
    show : bool, optional (default=True)
        If method is "box", show individual box plots for each feature.
    class_col : str or None, optional (default=None)
        Name of the class label column. If None, the function tries to infer 
        the class column by looking for a column with <10 unique values and 
        categorical/object/integer dtype.

    Returns:
    -------
    df : pandas.DataFrame
        The original dataframe passed in, unchanged.
    
    Raises:
    ------
    ValueError
        If class_col cannot be inferred or if method is invalid or insufficient 
        features are present for plotting.
    """
    df = df.drop(columns=['X','Y'])

    # Try to infer class column if not provided
    if class_col is None:
        for col in df.columns:
            if df[col].nunique() < 10 and df[col].dtype.name in [
                    'category', 'object', 'int64']:
                class_col = col
                break
        if class_col is None:
            raise ValueError(
                "Could not determine class column. Please specify 'class_col'.")

    # Filter numeric feature columns
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns\
                    if col != class_col]

    if method == "box":
        # Box plot interpretation:
        # - Box height (IQR): Spread of values
        # - Median: Line inside the box
        # - Whiskers: Range (excluding outliers)
        # - Outliers: Individual points
        # Use case:
        # - Feature distributions across classes
        # - Identify separable features
        
      for feat in feature_cols:
          if np.isfinite(df[feat]).all():
              # Remove outliers based on IQR
              Q1 = df[feat].quantile(0.1)
              Q3 = df[feat].quantile(0.9)
              IQR = Q3 - Q1
              lower_bound = Q1 - 1.1 * IQR
              upper_bound = Q3 + 1.1 * IQR
  
              df_filtered = df[(df[feat]>=lower_bound)&(df[feat]<=upper_bound)]
  
              # Plot
              plt.figure(figsize=(6, 4))
              sns.boxplot(x=class_col, y=feat, data=df_filtered, hue=class_col, 
                          palette='Set2', legend=False)
              plt.title(f'Distribution of {feat} across classes (no outliers)')
              plt.xlabel(class_col)
              plt.ylabel(feat)
              plt.tight_layout()
              plt.show()
          else:
              print(f"Skipping {feat} - contains inf/NaN.")


    elif method == "pair":
        # Pair plot interpretation:
        # - Each scatter plot shows how two features interact, color-coded b
        #   y class
        # - Diagonal shows feature distributions (histograms or KDEs)
        # Use case:
        # - Identify feature combinations that separate classes
        # - Spot clusters or trends

        if len(feature_cols) < 2:
            raise ValueError(
                "Need at least two numeric features to create a pairplot.")

        # Remove outliers across all selected features using IQR
        df_filtered = df.copy()
        for feat in feature_cols:
            if np.isfinite(df_filtered[feat]).all():
              # Remove outliers based on IQR
              Q1 = df[feat].quantile(0.1)
              Q3 = df[feat].quantile(0.9)
              IQR = Q3 - Q1
              lower_bound = Q1 - 1.1 * IQR
              upper_bound = Q3 + 1.1 * IQR
  
              df_filtered = df[(df[feat]>=lower_bound)&(df[feat]<=upper_bound)]
            else:
                print(f"Skipping outlier removal for {feat} (contains inf/NaN)")
    
        sns.pairplot(df_filtered[feature_cols + [class_col]], 
                     hue=class_col, 
                     palette='Set2')
        plt.suptitle("Feature pairwise relationships by class (no outliers)", 
                     y=1.02)
        plt.tight_layout()
        plt.show()


    elif method == "heat":
        # Heatmap interpretation:
        # - Shows correlation between all pairs of numeric features
        # - +1: Perfect positive correlation
        # -  0: No correlation
        # - -1: Perfect negative correlation
        # Use case:
        # - Detect redundant features (e.g., corr > 0.8)
        # - Feature reduction or PCA
        # - Clustered heatmap can reveal groups of related features

        feature_cols = df.select_dtypes(include=[np.number]).columns
        if class_col in feature_cols:
            feature_cols = feature_cols.drop(class_col)
        
        # Remove outliers from all numeric features using IQR
        df_filtered = df.copy()
        for feat in feature_cols:
            if np.isfinite(df_filtered[feat]).all():
                Q1 = df_filtered[feat].quantile(0.1)
                Q3 = df_filtered[feat].quantile(0.9)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.1 * IQR
                upper_bound = Q3 + 1.1 * IQR
                df_filtered = df_filtered[(df_filtered[feat] >= lower_bound) & 
                                          (df_filtered[feat] <= upper_bound)]
            else:
                print(f"Skipping outlier removal for {feat} (contains inf/NaN)")
            
        corr = df_filtered[feature_cols].corr()

        plt.figure(figsize=(max(8, 0.5 * len(feature_cols)), 
                            max(6, 0.4 * len(feature_cols))))
        sns.heatmap(corr, 
                    cmap='coolwarm', 
                    annot=True, 
                    fmt=".2f", 
                    square=True, 
                    cbar_kws={"shrink": .8})
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Unsupported method. Use 'box', 'pair', or 'heat'.")

