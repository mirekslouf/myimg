# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:19:46 2025

@author: p-sik
"""

from skimage.feature import match_template
from skimage import measure
import matplotlib.pyplot as plt
import myimg.apps.iLabels.roi as miroi
import numpy as np

def detector_correlation(image, mask, threshold=0.5, show=True):
    """
    Detect nanoparticles by correlating mask over image.

    Parameters:
        image: 2D np.array
            The input image where to detect nanoparticles.
        mask: 2D np.array
            The template mask (nanoparticle).
        threshold: float
            Minimum correlation score to consider a detection (default 0.5).

    Returns:
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