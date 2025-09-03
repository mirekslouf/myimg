# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:19:46 2025

@author: p-sik
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm  
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter
from skimage.feature import match_template
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
from scipy.ndimage import distance_transform_edt
from skimage.feature import match_template
from skimage import measure
import matplotlib.pyplot as plt
import myimg.apps.iLabels.roi as miroi
import numpy as np



def compute_local_variance(image, window_size=31):
    """
    Computes a local variance map of the input image using a uniform filter.
    
    Parameters
    ----------
    image : 2D array-like
        Grayscale input image (e.g., float32 or float64).
        
    window_size : int, optional
        Size of the sliding window over which variance is computed.
        Default is 31.
    
    Returns
    -------
    variance_map : 2D ndarray
        An image of the same shape as input, containing local variance values.
    
    Notes
    -----
    - Uses the formula Var(X) = E[X^2]-(E[X])^2 computed with uniform filtering.
    - Assumes input image is properly scaled (e.g., in [0, 1] or [0, 255]).
    """
    mean = uniform_filter(image, window_size)
    mean_sq = uniform_filter(image**2, window_size)
    return mean_sq - mean**2


def match_mask_in_image(image, mask, type_idx, threshold):
    """
    Applies template matching to detect regions in an image that resemble 
    a given mask.

    Parameters
    ----------
    image : 2D ndarray
        Input image (grayscale, float32 preferred).
        
    mask : 2D ndarray
        Template to match against the image (same type as image).
        
    type_idx : int
        Integer label identifying the type of template used.
        
    threshold : float
        Minimum normalized cross-correlation score to consider a match valid.

    Returns
    -------
    detections : list of dict
        Each detection is a dictionary with:
        - 'type': template type (type_idx)
        - 'x': x-coordinate of the detected region
        - 'y': y-coordinate
        - 'score': matching score at that location

    Notes
    -----
    - Uses `skimage.feature.match_template` for matching.
    - Uses a local maximum filter to avoid clustered duplicates.
    - Pad input enabled to detect near edges.
    """
    # Cross correlation of image and input mask
    result = match_template(image, mask, pad_input=True)

    mask_h, mask_w = mask.shape
    detections = []


    if np.max(result) < threshold:
        return []

    # Find maxima in the correlation map
    local_max = maximum_filter(result, 
                               size=(max(1, mask_h//15), 
                                     max(1, mask_w//15)))
    coordinates = np.argwhere((result==local_max) & (result > threshold))

    # Save the coordinates of detected nanoparticles
    for y, x in coordinates:
        detections.append({
            'type': type_idx,
            'x': x,
            'y': y,
            'score': result[y, x]
        })

    return detections


def remove_duplicates(detections, min_dist=10):
    """
    Removes duplicate detections by clustering spatially close points and 
    keeping only the highest-scoring detection in each cluster.
    
    Parameters
    ----------
    detections : list of dict
        List of detections, each as a dictionary with keys 'x', 'y', and 'score'.
        
    min_dist : float, optional
        Minimum distance (in pixels) to consider two detections as duplicates. 
        Default is 10.
    
    Returns
    -------
    unique_detections : list of dict
        Filtered list of detections, where only the best-scoring detection 
        per spatial cluster is retained.
    
    Notes
    -----
    - Uses DBSCAN clustering with `eps = min_dist` and `min_samples = 1`.
    - Assumes each detection dict has keys: 'x', 'y', and 'score'.
    """

    if not detections:
        return []

    coords = np.array([[d['x'], d['y']] for d in detections])
    scores = np.array([d['score'] for d in detections])

    # Cluster nearby detections
    clustering = DBSCAN(eps=min_dist, min_samples=1).fit(coords)
    labels = clustering.labels_

    # Pick the best scoring detection per cluster
    unique_detections = []
    for label in set(labels):
        indices = np.where(labels == label)[0]
        best_idx = indices[np.argmax(scores[indices])]
        unique_detections.append(detections[best_idx])

    return unique_detections


def detector_NCC(image, masks, threshold=0.6, show=True, 
                 n_jobs=-1, cmap="viridis", margin=30, ext=1.2):
    """
    Detects objects in an image using Normalized Cross-Correlation (NCC) 
    template matching, while masking out low-variance regions (typically 
    artefacts) to reduce false positives.

    Parameters
    ----------
    image : ndarray
        Input grayscale image as a 2D NumPy array.
        
    masks : list of ndarray
        List of template masks used for matching. Each mask is a 2D array.
        
    threshold : float, optional
        NCC score threshold for accepting detections.
        Default is 0.6.
        
    show : bool, optional
        If True, show visualizations of masked image and detections. 
        Default is True.
        
    n_jobs : int, optional
        Number of parallel jobs to run. 
        Default is -1 (use all cores).
        
    cmap : str, optional
        Colormap for displaying images. 
        Default is "viridis".
        
    margin : int, optional
        Minimum distance (in pixels) from image border to keep a detection. 
        Default is 30.
        
    ext : float, optional
        Extra scaling factor applied to minimum safe distance from low-variance 
        regions. Helps further reject detections near artefact edges.
        Default is 1.2.

    Returns
    -------
    output : pandas.DataFrame
        Table of detected coordinates and classes with columns:
        ['X', 'Y', 'Class', 'Note'].
        
    im_masked : ndarray
        Preprocessed image after low-variance masking

    Notes
    -----
    - Low-variance regions are assumed to correspond to artefacts or smooth 
      backgrounds.
    - The function automatically normalizes both the input image and masks 
      to [0, 1] range.
    - Duplicate detections are removed based on spatial proximity.
    - Detection filtering includes proximity to mask edges and artefact-adjacent 
      areas.
    """
    
    # Prepare image
    cut_bottom = 300
    height, width = image.shape[:2]
    image_cropped = image[:height - cut_bottom, :width]
    im = image_cropped.astype(np.float32)
    
    if im.max() > 2.0:
        im /= 255.0

    # Prepare normalized masks
    prepared_masks = []
    for idx, mask in enumerate(masks):
        m = mask.astype(np.float32)
        if m.max() > 2:
            m /= 255.0
        prepared_masks.append((idx, m))

    # Compute local variance mask
    variance_threshold = 0.01
    var_map = compute_local_variance(im, window_size=60)
    var_mask = var_map >= variance_threshold  # True for high-variance pixels
    low_var_mask = ~var_mask

    # Compute distance map from low-variance regions 
    dist_map = distance_transform_edt(~low_var_mask) 

    # Define min safe distance based on template size
    max_template_radius = max(max(m.shape) for _, m in prepared_masks) // 2
    min_safe_distance = max_template_radius + 60  

    # Mask out pixels too close to low-variance regions
    valid_area = dist_map >= min_safe_distance
    im_masked = im.copy()
    im_masked[~valid_area] = 0
    
    # final image used for detection
    im = im_masked  

    # Optional debug: Show masked image
    if show:
        plt.figure()
        plt.title("Masked Image")
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.show()

    # Run template matching
    if n_jobs == 1:
        results = [
            match_mask_in_image(im, m, idx, threshold)
            for idx, m in tqdm(prepared_masks, desc="Template Matching (debug)")
        ]
    else:
        tasks = [
            delayed(match_mask_in_image)(im, m, idx, threshold)
            for idx, m in prepared_masks
        ]
        
        results = list(tqdm(
                            Parallel(n_jobs=n_jobs)(tasks),
                            total=len(tasks),
                            desc="Template Matching"
                            ))

    detections = \
        [item for sublist in results if sublist is not None for item in sublist]


    # Remove detections too close to image borders
    h, w = im.shape[:2]
    filtered_detections = [
        d for d in detections
        if (margin <= d['x'] <= w - margin) and (margin <= d['y'] <= h - margin)
    ]
    
    # Remove duplicates
    filtered_detections = remove_duplicates(filtered_detections, min_dist=10)

    # Final filtering: only keep detections at a safe distance from mask
    final_detections = [
        d for d in filtered_detections
        if dist_map[int(d['y']), int(d['x'])] > min_safe_distance*ext
    ]

    # Plot final detections
    if show:
        plt.figure(figsize=(18, 9))
        plt.imshow(image, cmap=cmap)

        for idx, d in enumerate(final_detections):
            label = "nanoparticle" if idx == 0 else None
            plt.plot(d['x'], d['y'], 'x', color='red', label=label)

        if final_detections:
            plt.legend(loc='upper right')
        plt.title("Template Matching Detections")
        plt.axis('off')
        plt.show()
    
    # Prepare output DataFrame
    x_coords = [int(d['x']) for d in final_detections]
    y_coords = [int(d['y']) for d in final_detections]
    clss = [int(d['type'])+1 for d in final_detections]

    output = pd.DataFrame({
        "X": [round(x) for x in x_coords],
        "Y": [round(y) for y in y_coords],
        "Class": clss,
        "Note": None,
    })  
    
    return output, im_masked

def detector_correlation(image, mask, threshold=0.5, show=True):
    """
    Detect nanoparticles by correlating mask over image.

    Parameters
    ----------
    image: 2D np.array
        The input image where to detect nanoparticles.
        
    mask: 2D np.array
        The template mask (nanoparticle).
        
    threshold: float
        Minimum correlation score to consider a detection (default 0.5).

    Returns:
    --------
    centers: list of (row, col) tuples
        List of center coordinates of detected nanoparticles.
    """
    cut_bottom=300
    
    # Crop 
    height, width = image.shape[:2]
    image = image[:height - cut_bottom, :width]

    # # Preprocess image (expects NumPy array)
    # image = miroi.preprocess_image(image)
    
    # Step 1: Perform normalized cross-correlation
    correlation_map = match_template(image, mask, pad_input=True)

    # Step 2: Threshold the correlation map
    detected_peaks = (correlation_map >= threshold)

    # Step 3: Label connected regions
    labeled_peaks = measure.label(detected_peaks)

    # Step 4: Find center of each detected region
    regions = measure.regionprops(labeled_peaks)
    centers = []
    for region in regions:
        centers.append(region.centroid)  # (row, col)
        
    
    if show:
        plt.figure()
        plt.imshow(image, cmap='viridis')
        for (row, col) in centers:
            plt.plot(col, row, 'r+', markersize=10)

        plt.axis('off')
        plt.show()
        
    return centers

