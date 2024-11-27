# -*- coding: utf-8 -*-
<<<<<<< HEAD
=======
"""
Created on Tue Oct 22 12:16:25 2024

@author: Jakub
"""
# -*- coding: utf-8 -*-
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
'''
Interactive Plot for Particle Classification.
Created on: Oct 16, 2024
Author: Jakub
'''

import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< HEAD
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
=======
#import matplotlib.image as mpimg  # Required for image loading
#import numpy as np  # Required for image array handling
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1

# =============================================================================
# Level 1: Create plot with events and set classifier

def interactive_plot(im, ppar) -> tuple:
    '''
    Create an interactive plot for particle classification.

    Parameters
    ----------
<<<<<<< HEAD
    im : Image to be displayed in the plot.
    ppar : Object containing plot settings (xlabel, ylabel, xlim, ylim).

    Returns
    -------
    tuple : (figure, axes) for further manipulation.
    '''
    plt.close("all")  # Close all previous plots to prevent overlap
    initialize_interactive_plot_parameters()  # Initialize plot settings

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(num="Particle Classification")
    ax.imshow(im)  # Display the image

    # Set plot labels and limits
=======
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
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
    ax.set_xlabel(ppar.xlabel)
    ax.set_ylabel(ppar.ylabel)
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)

<<<<<<< HEAD
    # Initialize the particle classifier
    classifier = ParticleClassifier(ax=ax)

    # Show user instructions
    show_instructions()

    # Connect key press and close events to their handlers
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_keypress(event,\
                                                     ax, classifier, im, ppar))
=======
    # Initialize the particle classifier to manage particle data
    classifier = ParticleClassifier()

    # Show keyboard shortcuts for the user
    show_instructions()

    # Connect key press and close events to their respective handlers
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_keypress(event, ax, classifier)
    )
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
    fig.canvas.mpl_connect(
        "close_event", lambda event: on_close(event, ppar, classifier)
    )

    plt.tight_layout()  # Adjust layout for better spacing
<<<<<<< HEAD
    return fig, ax  # Return the created plot objects
=======
    return fig, ax  # Return the figure and axes objects
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1


def show_instructions():
    '''
<<<<<<< HEAD
    Display instructions for keyboard shortcuts used in classification.
=======
    Display instructions on how to use the keyboard shortcuts for classification.
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1

    Returns
    -------
    None
    '''
    print("\nInteractive Plot Instructions:")
<<<<<<< HEAD
    print(" - Press '1' to classify a particle as Class 1 \
          (Red-SS: Small Sharp).")
    print(" - Press '2' to classify a particle as Class 2 \
          (Blue-SB: Small Blurry).")
    print(" - Press '3' to classify a particle as Class 3 \
          (Green-BS: Big Sharp).")
    print(" - Press '4' to classify a particle as Class 4 \
          (Yellow-BB: Big Blurry).")
    print(" - Press '5' to save the current particle data.")
    print(" - Press '6' to remove the nearest particle marker.")
    print(" - Press 'q' to quit and close the plot.\n")

# =============================================================================
# Level 2: Callback functions for events

color_map = {  # Map classes to colors for visualization
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'yellow'
}


def del_bkg_point_close_to_mouse(classifier, x, y, ax, im, threshold=10):
    '''
    Delete the nearest particle point within a given threshold and redraw\
        the plot.

    Parameters
    ----------
    classifier : ParticleClassifier instance managing particle data.
    x, y : float : Coordinates of the mouse click.
    ax : matplotlib.axes.Axes : The matplotlib axis to redraw the plot on.
    im : numpy.ndarray : The image data to display as the background.
    threshold : int, optional : Distance threshold to consider a point for\
        removal.
=======
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
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1

    Returns
    -------
    None
    '''
<<<<<<< HEAD
    closest_index = None
    min_distance = float('inf')

    # Search for the closest particle
    for i, (px, py) in enumerate(zip(classifier.x_coords,
                                     classifier.y_coords)):
        distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if distance < min_distance and distance <= threshold:
            closest_index = i
            min_distance = distance

    if closest_index is not None:
        # Remove particle data from all related arrays
        removed_x = classifier.x_coords.pop(closest_index)
        removed_y = classifier.y_coords.pop(closest_index)
        removed_class = classifier.classes.pop(closest_index)
        classifier.plot_points.pop(closest_index)

        # Redraw the plot without the deleted point
        ax.clear()  # Clear the axes
        ax.imshow(im, origin="lower")  # Redraw the background image

        # Re-plot the remaining points
        for px, py, particle_class in zip(classifier.x_coords,
                                          classifier.y_coords, 
                                          classifier.classes):
            color = color_map.get(particle_class, 'black')
            ax.plot(px, py, '+', color=color, markersize=5)

        plt.draw()  # Redraw the plot
        print(f"Removed particle at ({removed_x:.2f}, {removed_y:.2f})\
              of class {removed_class}.")
    else:
        print("No particle found within threshold for deletion.")


def on_keypress(event, ax, classifier, im, ppar):
    '''
    Handle key press events for particle classification.

    Parameters
    ----------
    event : KeyEvent triggering the function.
    ax : Axes object for plotting.
    classifier : ParticleClassifier for managing data.
    im : numpy.ndarray : Background image for the plot.
    ppar : object containing output_file for saving.
    '''
    if event.key in ['1', '2', '3', '4']:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            particle_class = int(event.key)
            classifier.add_particle(x, y, particle_class)
            color = color_map.get(particle_class, 'black')
            ax.plot(x, y, color=color, markersize=5, marker='+')
            classifier.plot_points[-1] = ax.plot(x, y, '+', color=color,
                                                 markersize=5)[0]
            plt.draw()
            
    elif event.key == '5':  # Save all files
        # Save the particle data as "pdParticles.pkl"
        classifier.save_particles(filename="pdParticles.pkl")  
        print(f"Particle data saved to 'pdParticles.pkl'.")

        # Save coordinates to TXT with default output_file name
        df = classifier.get_coordinates()
        df.to_csv(f"{ppar.output_file}.txt", index=False)
        print(f"Coordinates saved to '{ppar.output_file}.txt'.")

        # Save the current plot to PNG with default output_file name
        plt.savefig(f"{ppar.output_file}.png")
        print(f"Plot saved as '{ppar.output_file}.png'.")

        print("All outputs saved successfully.")
        
    elif event.key == '6':
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            del_bkg_point_close_to_mouse(classifier, x, y, ax, im,
                                         threshold=10)
    elif event.key == 'q':  # Quit and save
        on_close(None, ppar, classifier)
        plt.close()
        print("Plot closed.")



def on_close(event, ppar, classifier):
    """
    Handle close event, saving all outputs.
    """
    # Save particle data
    classifier.save_particles(filename=f"{ppar.output_file}.pkl")
    print(f"Particle data saved to '{ppar.output_file}.pkl'.")

    # Save coordinates to TXT
    df = classifier.get_coordinates()
    df.to_csv(f"{ppar.output_file}.txt", index=False)
    print(f"Coordinates saved to '{ppar.output_file}.txt'.")

    # Save the plot as PNG
    plt.savefig(f"{ppar.output_file}.png")
    print(f"Plot saved as '{ppar.output_file}.png'.")

=======
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

>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
# =============================================================================
# Level 3: Particle Classifier Class

class ParticleClassifier:
<<<<<<< HEAD
    def __init__(self, ax=None):
        '''
        Initialize the classifier with empty data structures.

        Parameters
        ----------
        ax : Optional Axes object for plotting.
=======
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
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1

        Returns
        -------
        None
        '''
<<<<<<< HEAD
        self.x_coords = []
        self.y_coords = []
        self.classes = []
        self.notes = []
        self.class_labels = [1, 2, 3, 4]
=======
        self.x_coords = []  # Initialize x-coordinates list
        self.y_coords = []  # Initialize y-coordinates list
        self.classes = []    # Initialize classes list
        self.notes = []      # Initialize notes list

        self.class_labels = [1, 2, 3, 4]  # Define class labels
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
        self.class_notes = [
            "SS : Small Sharp",
            "SB : Small Blurry",
            "BS : Big Sharp",
            "BB : Big Blurry",
        ]
<<<<<<< HEAD
        self.output = pd.DataFrame()
        self.plot_points = []
        self.ax = ax if ax else plt.subplots()[1]

    def get_coordinates(self) -> pd.DataFrame:
        '''
        Create a DataFrame from stored particle data.

        Returns
        -------
        pd.DataFrame
        '''
        self.output = pd.DataFrame({
            "X": [round(x, 2) for x in self.x_coords],
            "Y": [round(y, 2) for y in self.y_coords],
            "Class": self.classes,
            "Note": self.notes,
        })
        return self.output

    def add_particle(self, x: float, y: float, particle_class: int) -> None:
        '''
        Add a particle with its data and plot it.

        Parameters
        ----------
        x : float, y : float : Coordinates of the particle.
        particle_class : int : Class of the particle (1-4).
=======
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
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1

        Returns
        -------
        None
        '''
<<<<<<< HEAD
        x = round(x, 2)  # Round X-coordinate to 2 decimal places
        y = round(y, 2)  # Round Y-coordinate to 2 decimal places 
        self.x_coords.append(x)
        self.y_coords.append(y)
        self.classes.append(self.class_labels[particle_class - 1])
        self.notes.append(self.class_notes[particle_class - 1])
    
        # Plot particle and save plot reference
        color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
        color = color_map.get(particle_class, 'black')
        plot_point, = self.ax.plot(x, y, '+', color=color, markersize=5)  
        self.plot_points.append(plot_point)  # Save the plot reference
    
        print(f"Added particle at (X={x}, Y={y}) as Class={particle_class}")

    def save_particles(self, filename: str = "pdParticles.pkl") -> None:
        self.get_coordinates().to_pickle(filename)
        print(f"Particles saved to '{filename}'.")

=======
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


>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
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
<<<<<<< HEAD
    An instance of DefaultParams containing default values for plot parameters.
=======
        An instance of DefaultParams containing default values for plot parameters.
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
    '''
    class DefaultParams:
        xlabel = "X-axis"  # Default x-axis label
        ylabel = "Y-axis"  # Default y-axis label
        xlim = [0, 1000]   # Default x-axis limits
        ylim = [0, 1000]   # Default y-axis limits
        output_file = "output"  # Default output file name
<<<<<<< HEAD
        pdParticles = "output"
=======

>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
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
<<<<<<< HEAD
        pdParticles = "output"
=======
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
        messages = True

    # Create and show the interactive plot with 'imParticles.png'
    ppar = MockPlotParams()  # Create mock plot parameters
<<<<<<< HEAD
    fig, ax = interactive_plot("imParticles.png", ppar) 
=======
    fig, ax = interactive_plot("imParticles.png", ppar)  # Initialize the interactive plot
>>>>>>> 519bdbd013c5e9cd003b5f518a8bb743a92fa1a1
    plt.show()  # Display the plot
