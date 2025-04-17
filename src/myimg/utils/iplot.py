# -*- coding: utf-8 -*-
'''
Interactive Plot for Particle Classification.
Created on: Oct 16, 2024
Author: Jakub
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# =============================================================================
# Level 1: Create plot with events and set classifier

def interactive_plot(im, ppar, filename="output.pkl", messages=False) -> tuple:
    '''
    Create an interactive plot for particle classification.
    '''
    plt.close("all")
    initialize_interactive_plot_parameters()

    # Create figure with 2-row grid: image (90%) + instructions (10%)
    fig = plt.figure(num="Particle Classification", figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
    
    # Main image in the top grid cell
    ax = fig.add_subplot(gs[0])
    ax.imshow(im)
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)
    ax.axis("off")

    # Text box in the bottom grid cell
    instruction_ax = fig.add_subplot(gs[1])
    instruction_ax.axis("off")  # Hide axes

    instructions = (
        "Press 1-4 to classify particles:\n"
        "  1: Red (Small Sharp) | 2: Blue (Small Blury) | 3: Green (Big Sharp) | 4: Purple (Big Blury)\n"
        "  5: Save | 6: Delete Nearest | r: Remove last | q: Quit"
    )

    instruction_ax.text(0.5, 0.5, instructions, ha='center', va='center', 
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    # Keep layout tidy
    plt.tight_layout()

    # Initialize classifier
    classifier = ParticleClassifier(ax=ax)
    
    if messages:
        show_instructions()

    fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_keypress(event, ax, classifier,
                                                     im, ppar, filename,
                                                     messages=messages))
    fig.canvas.mpl_connect(
        "close_event", lambda event: on_close(event, ppar, classifier,
                                              messages=messages))

    return fig, ax


def show_instructions():
    '''
    Display instructions for keyboard shortcuts used in classification.

    Returns
    -------
    None
    '''
    instructions = (
        "\nInteractive Plot Instructions:\n"
        " - Press '1' : Class 1 (Red-SS: Small Sharp).\n"
        " - Press '2' : Class 2 (Blue-SB: Small Blury).\n"
        " - Press '3' : Class 3 (Green-BS: Big Sharp).\n"
        " - Press '4' : Class 4 (Purple-BB: Big Blury).\n"
        " - Press '5' : Save data.\n"
        " - Press '6' : Remove the nearest particle marker.\n"
        " - Press 'r' : Remove last marker. \n"
        " - Press 'q' : Quit.\n"
    )


    print(instructions)
    
# =============================================================================
# Level 2: Callback functions for events

color_map = {  # Map classes to colors for visualization
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'purple'
}


def del_bkg_point_close_to_mouse(classifier, x, y, ax, im, threshold=10, messages=False):
    '''
    Delete the nearest particle point within a given threshold and redraw the plot.

    Parameters
    ----------
    classifier : ParticleClassifier instance managing particle data.
    x, y : float : Coordinates of the mouse click.
    ax : matplotlib.axes.Axes : The matplotlib axis to redraw the plot on.
    im : numpy.ndarray : The image data to display as the background.
    threshold : int, optional : Distance threshold to consider a point for removal.

    Returns
    -------
    None
    '''
    closest_index = None
    min_distance = float('inf')

    # Search for the closest particle
    for i, (px, py) in enumerate(zip(classifier.x_coords, classifier.y_coords)):
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
        ax.imshow(im)  # Redraw the background image

        # Re-plot the remaining points
        for px, py, particle_class in zip(classifier.x_coords, 
                                          classifier.y_coords, 
                                          classifier.classes):
            color = color_map.get(particle_class, 'black')
            ax.plot(px, py, '+', color=color, markersize=10)
        plt.axis("off")
        plt.draw()  # Redraw the plot
        if messages:
            print(f"Removed particle at ({removed_x:.2f}, {removed_y:.2f}) of class {removed_class}.")
    else:
        if messages:
            print("No particle found within threshold for deletion.")


def on_keypress(event, ax, classifier, im, ppar, filename="output", messages=False):
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
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}


    if event.key in ['1', '2', '3', '4']:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            particle_class = int(event.key)
            classifier.add_particle(x, y, particle_class, messages=messages)
            color = color_map.get(particle_class, 'black')
            ax.plot(x, y, color=color, markersize=10, marker='+')
            classifier.plot_points[-1] = ax.plot(x, y, '+', 
                                                 color=color, 
                                                 markersize=10)[0]
            plt.draw()
    elif event.key == '5': 
        # Save all files as pickle
        classifier.save_particles(filename=f"{filename}.pkl",
                                  messages=messages) 
        if messages:
            print(f"Particle data saved to '{filename}.pkl'.")

        # Save coordinates to TXT
        df = classifier.get_coordinates()
        df.to_csv(f"{filename}.txt", index=False)
        if messages:
            print(f"Coordinates saved to '{ppar.output_file}.txt'.")

        # Save the current plot to PNG
        plt.savefig(f"{filename}.png")
        if messages:
            print(f"Plot saved as '{filename}.png'.")

        print("All outputs saved successfully.")
        
    elif event.key == '6':
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            del_bkg_point_close_to_mouse(classifier, x, y, ax, im, threshold=10,
                                         messages=messages)
    elif event.key == 'q':  # Quit and save
        on_close(None, ppar, classifier, filename, messages=messages)
        plt.close()
        if messages:
            print("Plot closed.")
            
    elif event.key == 'r':
        if classifier.x_coords:
            classifier.x_coords.pop()
            classifier.y_coords.pop()
            classifier.classes.pop()
            classifier.notes.pop()
            point = classifier.plot_points.pop()
            point.remove()
            
            # Redraw the plot without the deleted point
            ax.clear()  # Clear the axes
            ax.imshow(im)  # Redraw the background image
            ax.axis('off')
            # Re-plot the remaining points
            for px, py, particle_class in zip(classifier.x_coords,
                                              classifier.y_coords, 
                                              classifier.classes):
                color = color_map.get(particle_class, 'black')
                ax.plot(px, py, '+', color=color, markersize=10)
           
            plt.draw()
 
            if messages:
                print("Last point removed.")



def on_close(event, ppar, classifier, filename="output", messages=False):
    """
    Handle close event, saving all outputs.
    """
    # Save particle data
    classifier.save_particles(filename=f"{filename}.pkl",
                              messages=messages)
    if messages:
        print(f"Particle data saved to '{filename}.pkl'.")

    # Save coordinates to TXT
    df = classifier.get_coordinates()
    df.to_csv(f"{filename}.txt", index=False)
    if messages:
        print(f"Coordinates saved to '{filename}.txt'.")

    # Save the plot as PNG
    plt.savefig(f"{filename}.png")
    if messages:
        print(f"Plot saved as '{filename}.png'.")

# =============================================================================
# Level 3: Particle Classifier Class

class ParticleClassifier:
    def __init__(self, ax=None, messages=False):
        '''
        Initialize the classifier with empty data structures.

        Parameters
        ----------
        ax : Optional Axes object for plotting.

        Returns
        -------
        None
        '''
        self.x_coords = []
        self.y_coords = []
        self.classes = []
        self.notes = []
        self.class_labels = [1, 2, 3, 4]
        self.class_notes = [
            "SS : Small Sharp",
            "SB : Small Blury",
            "BS : Big Sharp",
            "BB : Big Blury",
        ]
        self.output = pd.DataFrame()
        self.plot_points = []
        self.messages = messages
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

    def add_particle(self, x: float, y: float, particle_class: int, messages=False) -> None:
        '''
        Add a particle with its data and plot it.

        Parameters
        ----------
        x : float, y : float : Coordinates of the particle.
        particle_class : int : Class of the particle (1-4).

        Returns
        -------
        None
        '''
        x = round(x, 2)  # Round X-coordinate to 2 decimal places
        y = round(y, 2)  # Round Y-coordinate to 2 decimal places 
        self.x_coords.append(x)
        self.y_coords.append(y)
        self.classes.append(self.class_labels[particle_class - 1])
        self.notes.append(self.class_notes[particle_class - 1])
    
        # Plot particle and save plot reference
        color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
        color = color_map.get(particle_class, 'black')
        plot_point, = self.ax.plot(x, y, '+', color=color, markersize=10)  
        self.plot_points.append(plot_point)  # Save the plot reference
        if messages:
            print(f"Added particle at (X={x}, Y={y}) as Class={particle_class}")

    def save_particles(self, filename: str = "pdParticles.pkl", messages=False) -> None:
        self.get_coordinates().to_pickle(filename)
        if messages:
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
    plt.axis("off")
    plt.draw()


def default_plot_params(image):
    '''
    Provide default plot parameters based on the input image size.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The input image, used to determine default axis limits.

    Returns
    -------
    DefaultParams
        An instance of DefaultParams containing default values for plot 
        parameters.
    '''
    # Get image size from PIL or NumPy
    if hasattr(image, 'size'):  # PIL.Image
        width, height = image.size
    elif hasattr(image, 'shape'):  # np.ndarray
        height, width = image.shape[:2]
    else:
        raise TypeError("Unsupported image type. Provide a PIL.Image or NumPy array.")

    class DefaultParams:
        xlim = [0, width]
        ylim = [height, 0]
        output_file = "output"
        pdParticles = "particles"  # Add this line

    return DefaultParams()


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
        pdParticles = "output"
        messages = True

    # Create and show the interactive plot with 'imParticles.png'
    ppar = MockPlotParams()  # Create mock plot parameters
    fig, ax = interactive_plot("imParticles.png", ppar) 
    plt.show()  # Display the plot
