'''
Module: myimg.apps.MDistr.plot
------------------------------

Plot pre-calculated distributions/histograms.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def histogram(data, bins, normalize=None, fitting=None,
              color1='blue', color2='red', width1=0.8, width2=1, alpha=0.5,
              xlabel=None, xlim=None, xticks=None, mxticks=None, 
              ylabel=None, ylim=None, yticks=None, myticks=None,
              legend=None, grid=None,
              out_file=None, out_dpi=300):

    # TODO: provisional version
    # to be decomposed and made more flexible

    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins)
    
    # Calculate bin_widths
    # (bin_widths are used to plot histogram by means of plt.bar
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate bin_centers
    # (bin_centers are needed to fit function to dta
    # (bin_centers = average between the bins: 1st/2nd, 2nd/3rd ...
    # (trick: bin_edges[:1] ~ all but the last + bin_edges[1:] ~ all but the 1st 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Gaussian function
    def gaussian(x , amplitude, mu, sigma):
        return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    # Fit Gaussian to data if requested
    if fitting == 'gaussian':
        if normalize is None:
            iguess = [np.max(counts), np.mean(data), np.std(data)]
            par,cov = curve_fit(gaussian, bin_centers, counts, p0=iguess)
            amplitude, mu, sigma = par
    
    # Prepare X-data to plot Gaussian
    X = np.linspace(bin_edges[0], bin_edges[-1], 501, endpoint=True)
    
    # Plot
    # (a) Plot histogram
    plt.bar(bin_centers, counts, 
            color=color1, width=bin_width*width1, alpha=alpha, 
            label="Counts")
    # (b) Plot the fit
    if fitting == 'gaussian':
        plt.plot(X, gaussian(X, amplitude, mu, sigma), 
                 color=color2, lw=width2, label="Gaussian fit")
    # (c) Additional params
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if xticks is not None:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))    
    if yticks is not None:
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(yticks))
    if mxticks is not None:
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(mxticks))    
    if myticks is not None:
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(myticks))     
    # (d) Finalization
    if legend is True: plt.legend()
    plt.tight_layout()
    # (e) Save (if requested) and show
    if out_file is not None:
        plt.savefig(out_file, dpi=out_dpi)
    plt.show()
