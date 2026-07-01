'''
Module: myimg.apps.MDistr.plot
------------------------------

Plot pre-calculated distributions/histograms.
'''

import myimg.api as mi

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Plots(mi.Plots):

    @staticmethod
    def histogram(df, distribution, xlabel=None, ylabel=None, ax=None, 
                  title=None, width=0.6, color="C0", grid=False,
                  xlim=None, xtics=None, mxtics=None,
                  ylim=None, ytics=None, mytics=None,
                  out_file=None, out_dpi=300):

        # Create new fig,ax => if ax argument was not specified
        if ax is None: fig, ax = plt.subplots()

        # Define X,Y values        
        x = df["Center"].to_numpy(dtype=float)
        y = df[distribution].to_numpy(dtype=float)

        # Basic plot (zorder to get bars in front of optional grid)
        ax.bar(x, y, width=width, color=color, zorder=100)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Adjust X-axis
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtics))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(mxtics))
        
        # Adjust Y-axis
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(plt.MultipleLocator(ytics))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(mytics))
        
        # Final adjustments
        if grid is True: ax.grid(axis="y")
        fig.tight_layout()
        
        # Save file if requested
        if out_file is not None:
            fig.savefig(out_file, dpi=out_dpi)

        # Return the plot/axes
        # (this will also draw the figure in Spyder
        return ax


    @staticmethod
    def histogram_NV(df, title=None, ax=None,
                     ptype='bar', width=0.35, shift=0.05,
                     colors=("C0", "C1"), grid=False,
                     xlabel=None, xlim=None, xtics=None, mxtics=None,
                     ylabel=None, ylim=None, ytics=None, mytics=None,
                     out_file=None, out_dpi=300):

        # Create new fig, ax if not provided
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            
        # Calculate bin_width, width, shift
        # (width and shift parameters should be relative to bin_width
        bin_width = df.Min[2]-df.Min[1]
        width = bin_width * width
        shift = bin_width * shift

        # X values
        x = df["Center"].to_numpy(dtype=float)

        # Plot
        if ptype == 'bar':
            ax.bar(x - width/2 - shift, df["N%"].to_numpy(dtype=float),
               width=width, color=colors[0], label="2dN%", zorder=100)
            ax.bar(x + width/2 + shift, df["V%"].to_numpy(dtype=float),
               width=width, color=colors[1], label="2dV%", zorder=100)
        elif ptype == 'vlines':
            ax.vlines(x - shift, 0, df["N%"].to_numpy(dtype=float),
               linewidth=width, color=colors[0], label="N%", zorder=100)
            ax.vlines(x + shift, 0, df["V%"].to_numpy(dtype=float),
               linewidth=width, color=colors[1], label="V%", zorder=100)

        # Labels / title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # X-axis control
        ax.set_xlim(xlim)
        if xtics is not None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(xtics))
        if mxtics is not None:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(mxtics))

        # Y-axis control
        ax.set_ylim(ylim)
        if ytics is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(ytics))
        if mytics is not None:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(mytics))

        # Grid
        if grid is True:
            ax.grid(axis="y")

        # Legend (important difference vs single histogram)
        ax.legend(handlelength=0.8)

        # Layout
        fig.tight_layout()

        # Save if requested
        if out_file is not None:
            fig.savefig(out_file, dpi=out_dpi)

        return ax

class Templates:


    def hist_with_gaussian(data, bins, normalize=None, fitting=None,
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
        # (bin_centers are needed to fit function to data
        # (bin_centers = average between the bins: 1st/2nd, 2nd/3rd ...
        # (trick: bin_edges[:-1] ~ all but the last
        # (       bin_edges[1:] ~ all but the 1st 
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
