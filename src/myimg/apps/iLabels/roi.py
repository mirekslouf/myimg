# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 07:10:29 2025

@author: p-sik
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import myimg.api as mi
import matplotlib.patches as patches
import random
import os
import pickle
import math



def load_myimg(image, cut_bottom=300, save_as="output.tif", show=False):
    """
    Load an image from a file path or PIL.Image, crop pixels from the bottom, 
    preprocess, save the result, and return both the original and preprocessed 
    cropped images as NumPy arrays.
    
    Parameters:
    -----------
    image : str or PIL.Image.Image
        File path or PIL image object.
    cut_bottom : int
        Number of pixels to cut from the bottom.
    save_as : str
        Filename to save the cropped image.
    show : bool
        Whether to display the cropped image.
    
    Returns:
    --------
    tuple: (original_image_array, cropped_preprocessed_image_array)
    """
    
    # Load image
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("Input must be a file path / PIL.Image.Image object.")
        
    img_np = np.array(img)
    
    width, height = img.size

    # Crop using PIL
    cropped = img.crop((0, 0, width, height - cut_bottom))
    cropped_np = np.array(cropped)

    # Preprocess image (expects NumPy array)
    preprocc = preprocess_image(cropped_np)

    # Save preprocessed image
    Image.fromarray(preprocc).save(save_as)

    # Optionally show image
    if show:
        plt.figure()
        plt.imshow(preprocc, cmap="viridis")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return img_np, preprocc


def preprocess_image(image, apply_clahe=True, gamma=1.2, normalize=True):
    """
    Preprocess SEM/TEM image to correct for contrast and brightness variations.

    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR, RGB, or grayscale).
    apply_clahe : bool
        Whether to apply CLAHE for local contrast enhancement.
    gamma : float
        Gamma value for gamma correction (1.0 = no change).
    normalize : bool
        Whether to normalize intensity values to 0–255.

    Returns:
    gamma_corrected : np.ndarray
        Preprocessed image (grayscale, uint8).
    """

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure grayscale image is uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i/255.0)**inv_gamma*255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    # Normalize intensity to full 8-bit range if needed
    if normalize:
        norm_img = cv2.normalize(gamma_corrected, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return norm_img

    return gamma_corrected


def prep_data(fpath_img, fpath_peaks, min_xy=20, imID='im01', show=False):
    """
    Load an image and its peak annotations, and filter out low-X/Y points.

    Parameters:
    -----------
    fpath_img : str
        Path to the image file (e.g., .tif).
    fpath_peaks : str
        Path to the associated peaks file (e.g., .pkl).
    min_xy : int, optional
        Minimum allowed value for X and Y coordinates (default is 20).
    imID : str, optional
        Identifier to assign to all peaks (default is 'im01').
    show : bool, optional
        If True, shows the annotated image using `show_in_image`.
   
    Returns:
    --------
    arr : np.ndarray
        Image array.
    df : pd.DataFrame
        Filtered DataFrame with peak coordinates.
    img_obj : MyImage
        The loaded MyImage object.
    """
    # Load image and convert it to numpy.array
    img_obj = mi.MyImage(fpath_img, peaks=True)
    arr = np.array(img_obj.img)
   
    # Load coordinates of peaks from the image
    img_obj.peaks.read(fpath_peaks)
    
    # Optionally show image with detected peaks
    if show:
        img_obj.peaks.show_in_image()
    
    # Extract dataframe with coordinates
    df = img_obj.peaks.df
    
    # Avoid peaks of which a ROI cannot be created
    df = df[(df.X > min_xy) & (df.Y > min_xy)].reset_index(drop=True)
    
    # Assign data with an image ID
    df['imID'] = imID    
    
    return arr, df, img_obj


def get_ROIs(im, df, s=20, norm=True, show=False):
    """
    Extract and visualize square ROIs (Regions of Interest) from an image 
    based on coordinates provided in a DataFrame. Each ROI is centered 
    on the brightest region (maximum intensity) within a preliminary square 
    extracted around the initial coordinate.

    If the center shifts, the function updates the coordinates in the returned 
    DataFrame.

    Parameters:
    -----------
    im : np.ndarray
        Input 2D (grayscale) image.

    df : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'X': horizontal coordinate (column index).
        - 'Y': vertical coordinate (row index).
        - 'Class': class label, used for color coding in visualization.

    s : int, optional
        Half-side size of each ROI (ROI will be of size (2*s) x (2*s)).
        Default is 20.

    norm : bool, optional
        If True, return also min-max normalized versions of the ROIs.
        Default is True.
    
    show :  bool, optional
        If True, the input image will be displayed with detected ROIs.

    Returns:
    --------
    arr_roi : list of np.ndarray
        List of extracted ROIs (centered on the brightest region).

    arr_norm : list of np.ndarray
        List of min-max normalized ROIs (only if norm=True).

    corrected_df : pandas.DataFrame
        Updated DataFrame with corrected 'X' and 'Y' coordinates reflecting 
        the center of maximum intensity inside each ROI.

    Notes:
    ------
    - If an initial ROI around a coordinate would go out of image bounds, 
      the ROI is skipped and a warning is printed.
    - If the re-centered ROI would go out of bounds, fallback to original 
      preliminary ROI.
    - The function visualizes the original image with ROI rectangles drawn on
      it, color-coded according to the class label.
    """
    
    # Initialize output variables    
    arr_roi = []
    corrected_df = df.copy()
    
    # Color map for visualization
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}

    # For legend proxy patches
    proxy_patches = {} 
    
    # Set up figure if show=True
    if show:
        fig, ax = plt.subplots()
        ax.imshow(im, cmap='viridis')
        
    # Image dimensions
    h, w = im.shape 
    
    # Indices of ROIs that were successfully extracted
    valid_indices = []                

    for i in range(len(df)):
        # Center coordinates
        x = int(df.Y[i])  # row
        y = int(df.X[i])  # column
        
        # Corresponding class
        clss = df.Class[i]
        
        # Default to black if unknown class
        color = color_map.get(clss, 'black')
    
        # Check if preliminary ROI would go out of image bounds
        if x - s < 0 or x + s > h or y - s < 0 or y + s > w:
            print(f"Skipping index {i}: original ROI out of bounds.")
            continue
    
        # Extract preliminary ROI
        roi_prelim = im[x - s:x + s, y - s:y + s]
    
        # Find coordinates of the maximum intensity within the ROI
        max_val = roi_prelim.max()
        coords = np.column_stack(np.where(roi_prelim == max_val))
        center_rel_y, center_rel_x = np.mean(coords, axis=0)
        center_rel_y, center_rel_x = \
            int(round(center_rel_y)), int(round(center_rel_x))
            
        # Convert local maximum coordinates to absolute image coordinates
        max_abs_y = x - s + center_rel_y
        max_abs_x = y - s + center_rel_x
    
        # Check if the new, centered ROI stays within bounds
        if (max_abs_y - s >= 0 and max_abs_y + s <= h and
            max_abs_x - s >= 0 and max_abs_x + s <= w):
            roi = im[max_abs_y - s:max_abs_y + s, max_abs_x - s:max_abs_x + s]
            rect_x, rect_y = max_abs_x, max_abs_y
    
            # Update corrected coordinates
            corrected_df.at[i, 'X'] = max_abs_x
            corrected_df.at[i, 'Y'] = max_abs_y
        else:
            # Fall back to the preliminary ROI if centered ROI is out of bounds
            roi = roi_prelim
            rect_x, rect_y = y, x
    
        # Save ROI and corresponding index
        arr_roi.append(roi)
        valid_indices.append(i)
    
        # Draw rectangle on the image if show=True
        if show:
            rect = patches.Rectangle((rect_x - s, rect_y - s), 
                                     2 * s, 2 * s,
                                     linewidth=1.5, 
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add proxy patch if not already added
            if clss not in proxy_patches:
                proxy_patches[clss] = patches.Patch(color=color, 
                                                    label=f'Class {clss}')  
                
    
    # Set up plot appearance if show=True    
    if show:
        ax.legend(handles=list(proxy_patches.values()), loc='upper right') 
        ax.set_title("Original image with ROI rectangles")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    
    # Keep only valid (non-skipped) entries
    corrected_df = corrected_df.loc[valid_indices].reset_index(drop=True)

    # Optionally normalize ROIs
    if norm:
        arr_norm = min_max_normalize(arr_roi)
        return arr_roi, arr_norm, corrected_df
    else:
        return arr_roi, corrected_df


def create_masks(rois, df, class_col="class", n_per_class=10, show=True, save=False, save_path="."):
    """
    Extracts n ROIs per class and computes average masks for each class.
    
    Parameters:
    -----------
    rois : list of tuple
        List of ROI tuples (roi, label). Only the first element (roi) is used.
        
    df : pd.DataFrame
        DataFrame containing labels for ROIs (must match rois list length).
    
    class_col : str
        Column in df that contains class labels.
        
    n_per_class : int
        Number of samples per class to average.
    
    show : bool
        If True, display the average masks per class.

    save : bool
        If True, save each average mask as a .pkl file.

    save_path : str
        Path to the folder where the mask files will be saved.
    
    Returns:
    --------
    mean_masks : dict
        Dictionary mapping class labels to their average mask (mean of n ROIs).
        
    class_order : list
        List of class labels in the order they are plotted.
    """

    # Extract only the image arrays if rois is a list of (roi, label) tuples
    rois_only = [roi_data[0] for roi_data in rois]

    df = df.reset_index(drop=True)
    rois_only = np.array(rois_only)
    
    mean_masks = {}
    class_order = sorted(df[class_col].unique())

    if show:
        fig, axs = plt.subplots(1, len(class_order), 
                                figsize=(3 * len(class_order), 3))
        if len(class_order) == 1:
            axs = [axs]  # Ensure it's iterable

    for i, lbl in enumerate(class_order):
        # Get indices for this class
        class_indices = df[df[class_col] == lbl].index.tolist()
        
        if len(class_indices) < n_per_class:
            print(f"Warning: class '{lbl}' has only {len(class_indices)} samples.")
            selected_indices = class_indices
        else:
            selected_indices = np.random.choice(class_indices,
                                                n_per_class, 
                                                replace=False)
        
        roi_stack = np.stack([rois_only[i] for i in selected_indices])
        mean_mask = np.mean(roi_stack, axis=0)
        mean_masks[lbl] = mean_mask

        if save:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"mask{i+1}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(mean_mask, f)

        if show:
            axs[i].imshow(mean_mask, cmap="viridis")
            axs[i].set_title(f"Class {lbl}")
            axs[i].axis("off")

    if show:
        plt.tight_layout()
        plt.show()

    return mean_masks, class_order
        

def min_max_normalize(data, mrange=255):
    """
    Normalizes the list of images (2D arrays) using min-max scaling, rescaling 
    the values to a specified range for each individual image.
    
    Parameters
    ----------
    data : list of np.ndarray
        A list where each element is a 2D numpy array representing an image.
    
    mrange : int, optional
        The desired range for normalization. Default is 255, which scales 
        the data between 0 and 255.
    
    Returns
    -------
    normalized_data : list of np.ndarray
        A list where each element is a normalized 2D numpy array (image).
    """
    # Initialize output variables
    normalized_data = []
    
    for image in data:
        # Ensure the image is a numpy array
        image = np.array(image)
        
        # Compute min and max of the image
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Avoid division by zero if all values in the image are the same
        if min_val == max_val:
            # if all values are the same, set it to 0
            nimage = np.zeros_like(image)  
        else:
            # Normalize the image
            nimage = (image - min_val) / (max_val - min_val) * mrange 
        
        normalized_data.append(nimage)
    
    return normalized_data


def show_random_rois(rois, df, n=5, cmap="viridis"):
    """
    Display a random subset of ROI images as subplots in one figure.

    Parameters:
    -----------
    rois : list of tuple
        Each element is (roi, label) where:
        - roi : np.ndarray, the ROI image
        - label : identifier (str or int)

    df : pandas.DataFrame
        Metadata about the ROIs. Must match `rois` in length.
        If 'Note' exists, it’s used in titles.

    n : int, optional
        Number of ROIs to show. Default is 5.

    cmap : str, optional
        Colormap for displaying images.
    """
    assert len(rois) == len(df), "Length of ROIs and DataFrame must match."
    n = min(n, len(rois))

    indices = random.sample(range(len(rois)), n)

    # Layout: always 2 rows
    ncols = math.ceil(n / 2)
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, idx in zip(axes, indices):
        roi, label = rois[idx]
        ax.imshow(roi, cmap=cmap)
        title = f"{df.Note.iloc[idx]} (Image: {label})" \
            if 'Note' in df.columns else f"ROI {idx} ({label})"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()