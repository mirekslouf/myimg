# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:16:25 2024

@author: Jakub
"""
# -*- coding: utf-8 -*-
'''
Interactive Plot for Particle Classification.
Created on: Oct 16, 2024
Author: Jakub
'''

import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg  # Required for image loading
#import numpy as np  # Required for image array handling
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output


# =============================================================================
# Level 1: Create plot with events and set classifier

def interactive_plot(im, ppar) -> tuple:
    '''
    Create an interactive plot for particle classification.

    Parameters
    ----------
    im :  image itself 

    ppar : object
        Parameter object containing plot settings (xlabel, ylabel, xlim, ylim).

    Returns
    -------
    tuple
        A tuple containing the figure and axes objects for further manipulation.
    '''
    plt.close("all")  # Ensure all previous plots are closed
    initialize_interactive_plot_parameters()  # Set plot parameters
    
    # Set up the plot and load the background image
    fig, ax = plt.subplots(num="Particle Classification")
    ax.imshow(im)  # Display the image

    # Set plot labels and limits based on the provided parameters
    ax.set_xlabel(ppar.xlabel)
    ax.set_ylabel(ppar.ylabel)
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)

    # Initialize the particle classifier to manage particle data
    classifier = ParticleClassifier()

    # Show keyboard shortcuts for the user
    show_instructions()

    # Connect key press and close events to their respective handlers
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_keypress(event, ax, classifier)
    )
    fig.canvas.mpl_connect(
        "close_event", lambda event: on_close(event, ppar, classifier)
    )

    plt.tight_layout()  # Adjust layout for better spacing
    return fig, ax  # Return the figure and axes objects


def show_instructions():
    '''
    Display instructions on how to use the keyboard shortcuts for classification.

    Returns
    -------
    None
    '''
    print("\nInteractive Plot Instructions:")
    print(" - Press '1' to classify a particle as Class 1 (Red-SS : Small Sharp).")
    print(" - Press '2' to classify a particle as Class 2 (Blue-SB : Small Blurry).")
    print(" - Press '3' to classify a particle as Class 3 (Green-BS : Big Sharp).")
    print(" - Press '4' to classify a particle as Class 4 (Yellow-BB : BigBlurry).")
    print(" - Press '5' to save the current particle data.")
    print(" - Press 'q' to quit the interactive session and close the plot.\n")


# =============================================================================
# Level 2: Callback functions for events

# Define a color map for each class to visualize particles
color_map = {
    1: 'red',      # Class 1 -> Red
    2: 'blue',     # Class 2 -> Blue
    3: 'green',    # Class 3 -> Green
    4: 'yellow'    # Class 4 -> Yellow
}

def on_keypress(event, ax, classifier):
    '''
    Handle key press events for particle classification.

    Parameters
    ----------
    event : matplotlib.backend_bases.KeyEvent
        The key press event that triggered this function.
    ax : matplotlib.axes.Axes
        The axes object to plot the particles on.
    classifier : ParticleClassifier
        The classifier managing the particle data.

    Returns
    -------
    None
    '''
    if event.key in ['1', '2', '3', '4']:  # Check if a class key is pressed
        # Get the current mouse position in the plot coordinates
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:  # Ensure the coordinates are valid
            # Add the particle with the selected class
            particle_class = int(event.key)
            classifier.add_particle(x, y, particle_class)  # Store particle data

            # Use the appropriate color from the color_map
            color = color_map.get(particle_class, 'black')  # Default to black if class not found

            # Plot the particle with the selected color
            ax.plot(x, y, 'o', color=color, markersize=5)  # Circle marker with the class color

            plt.draw()  # Redraw the plot to update with the new point

    elif event.key == '5':  # If '5' is pressed, save particle data
        classifier.save_particles()  # Save the particles
        print("Particle data saved to PKL and TXT.")

    elif event.key == 'q':  # If 'q' is pressed, quit the interactive session
        plt.close()  # Close the plot
        print("Plot closed.")


def on_close(event, ppar, classifier):
    '''
    Handle the close event and save results.

    Parameters
    ----------
    event : matplotlib.backend_bases.CloseEvent
        The close event that triggered this function.
    ppar : object
        Parameter object containing output file information.
    classifier : ParticleClassifier
        The classifier managing the particle data.

    Returns
    -------
    None
    '''
    # Save the plot as a PNG file
    plt.savefig(f"{ppar.output_file}.png")
    print(f"Plot saved as '{ppar.output_file}.png'.")

    # Save particle data to a TXT file
    df = classifier.get_coordinates()  # Get the particle coordinates
    df.to_csv(f"{ppar.output_file}.txt", index=False)  # Save to CSV format
    print(f"Coordinates saved to '{ppar.output_file}.txt'.")

    # Confirm all outputs
    print("Interactive plot closed. Output files:")
    print(f" - {ppar.output_file}.txt")
    print(f" - {ppar.output_file}.pkl")
    print(f" - {ppar.output_file}.png")

# =============================================================================
# Level 3: Particle Classifier Class

class ParticleClassifier:
    '''
    Class to store and manage particle data.

    Attributes
    ----------
    x_coords : list
        List to store x-coordinates of the particles.
    y_coords : list
        List to store y-coordinates of the particles.
    classes : list
        List to store the classes of the particles.
    notes : list
        List to store notes associated with the particle classes.
    class_labels : list
        List of labels corresponding to each particle class.
    class_notes : list
        List of notes corresponding to each particle class.
    output : pandas.DataFrame
        DataFrame to store the output particle data.

    Methods
    -------
    get_coordinates() -> pd.DataFrame
        Return particle data as a DataFrame.
    add_particle(x: float, y: float, particle_class: int) -> None
        Add a particle and store its data.
    save_particles(filename: str = "pdParticles.pkl") -> None
        Save particle data to a pickle file.
    '''

    def __init__(self):
        '''
        Initialize ParticleClassifier object with empty data structures.

        Returns
        -------
        None
        '''
        self.x_coords = []  # Initialize x-coordinates list
        self.y_coords = []  # Initialize y-coordinates list
        self.classes = []    # Initialize classes list
        self.notes = []      # Initialize notes list

        self.class_labels = [1, 2, 3, 4]  # Define class labels
        self.class_notes = [
            "SS : Small Sharp",
            "SB : Small Blurry",
            "BS : Big Sharp",
            "BB : Big Blurry",
        ]
        self.output = pd.DataFrame()  # Initialize DataFrame for output

    def get_coordinates(self) -> pd.DataFrame:
        '''
        Return particle data as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing particle coordinates and associated data.
        '''
        self.output = pd.DataFrame({
            "X": self.x_coords,
            "Y": self.y_coords,
            "Class": self.classes,
            "Note": self.notes,
        })
        return self.output  # Return the DataFrame

    def add_particle(self, x: float, y: float, particle_class: int) -> None:
        '''
        Add a particle and store its data.

        Parameters
        ----------
        x : float
            The x-coordinate of the particle.
        y : float
            The y-coordinate of the particle.
        particle_class : int
            The class of the particle (1-4).

        Returns
        -------
        None
        '''
        self.x_coords.append(x)  # Store the x-coordinate
        self.y_coords.append(y)  # Store the y-coordinate
        self.classes.append(self.class_labels[particle_class - 1])  # Store the class
        self.notes.append(self.class_notes[particle_class - 1])  # Store the corresponding note
        print(f"Added particle at (X={x}, Y={y}) as Class={particle_class}")

    def save_particles(self, filename: str = "pdParticles.pkl") -> None:
        '''
        Save particle data to a pickle file.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the particle data to (default is "pdParticles.pkl").

        Returns
        -------
        None
        '''
        self.get_coordinates().to_pickle(filename)  # Save the coordinates DataFrame to a pickle file
        print(f"Particles saved to '{filename}'.")


# =============================================================================
# Level 4: Auxiliary Functions

def initialize_interactive_plot_parameters():
    '''
    Initialize parameters for the interactive plot.

    Returns
    -------
    None
    '''
    plt.rcParams.update({
        'figure.figsize': (6, 4),  # Set default figure size
        'figure.dpi': 100,          # Set figure resolution
        'font.size': 12,            # Set default font size
        'lines.linewidth': 1.0       # Set default line width
    })


def clear_plot():
    '''
    Clear the interactive plot while keeping labels and limits.

    Returns
    -------
    None
    '''
    ax = plt.gca()  # Get the current axes
    xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()  # Store current labels
    xlim, ylim = plt.xlim(), plt.ylim()  # Store current limits
    plt.cla()  # Clear the axes
    plt.xlabel(xlabel)  # Restore x-label
    plt.ylabel(ylabel)  # Restore y-label
    plt.xlim(xlim)  # Restore x-limits
    plt.ylim(ylim)  # Restore y-limits

def default_plot_params():
    '''
    Provide default plot parameters if none are given.

    Returns
    -------
    DefaultParams
        An instance of DefaultParams containing default values for plot parameters.
    '''
    class DefaultParams:
        xlabel = "X-axis"  # Default x-axis label
        ylabel = "Y-axis"  # Default y-axis label
        xlim = [0, 1000]   # Default x-axis limits
        ylim = [0, 1000]   # Default y-axis limits
        output_file = "output"  # Default output file name

    return DefaultParams()  # Return an instance of DefaultParams


# =============================================================================
# Example Usage

if __name__ == "__main__":
    class MockPlotParams:
        '''
        Mock object to simulate plot parameters for demonstration purposes.

        Attributes
        ----------
        xlabel : str
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.
        xlim : list
            Limits for the x-axis.
        ylim : list
            Limits for the y-axis.
        output_file : str
            Base name for output files.
        messages : bool
            Flag to control message printing (not used in this mock).
        '''
        xlabel = "X-axis"
        ylabel = "Y-axis"
        xlim = [0, 1000]
        ylim = [0, 1000]
        output_file = "output"
        messages = True

    # Create and show the interactive plot with 'imParticles.png'
    ppar = MockPlotParams()  # Create mock plot parameters
    fig, ax = interactive_plot("imParticles.png", ppar)  # Initialize the interactive plot
    plt.show()  # Display the plot
