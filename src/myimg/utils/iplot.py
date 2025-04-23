# -*- coding: utf-8 -*-
'''
Interactive Plot for Particle Classification.
Created on: Oct 16, 2024
Author: Jakub
'''

import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# =============================================================================
# Constants and Configurations

COLOR_MAP = {
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'yellow'
}

CLASS_NOTES = {
    1: "SS : Small Sharp",
    2: "SB : Small Blurry",
    3: "BS : Big Sharp",
    4: "BB : Big Blurry"
}

# =============================================================================
# Level 1: Create plot with events and set classifier

def interactive_plot(im, ppar) -> tuple:
    plt.close("all")
    initialize_interactive_plot_parameters()

    fig, ax = plt.subplots(num="Particle Classification")
    ax.imshow(im)
    ax.set_xlabel(ppar.xlabel)
    ax.set_ylabel(ppar.ylabel)
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)

    classifier = ParticleClassifier(ax=ax)
    show_instructions()

    fig.canvas.mpl_connect("key_press_event", lambda event: on_keypress(event, ax, classifier, im, ppar))
    fig.canvas.mpl_connect("close_event", lambda event: on_close(event, ppar, classifier))

    plt.tight_layout()
    return fig, ax


def show_instructions():
    print("\nInteractive Plot Instructions:")
    for i in range(1, 5):
        print(f" - Press '{i}' to classify a particle as Class {i} ({CLASS_NOTES[i]}).")
    print(" - Press '5' to save the current particle data.")
    print(" - Press '6' to remove the nearest particle marker.")
    print(" - Press 'q' to quit and close the plot.\n")


# =============================================================================
# Level 2: Callback functions

def del_bkg_point_close_to_mouse(classifier, x, y, ax, im, threshold=10):
    closest_index = None
    min_distance = float('inf')

    for i, (px, py) in enumerate(zip(classifier.x_coords, classifier.y_coords)):
        distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if distance < min_distance and distance <= threshold:
            closest_index = i
            min_distance = distance

    if closest_index is not None:
        removed_x = classifier.x_coords.pop(closest_index)
        removed_y = classifier.y_coords.pop(closest_index)
        removed_class = classifier.classes.pop(closest_index)
        removed_note = classifier.notes.pop(closest_index)
        classifier.plot_points.pop(closest_index)

        ax.clear()
        ax.imshow(im, origin="lower")
        ax.set_xlabel(classifier.ax.get_xlabel())
        ax.set_ylabel(classifier.ax.get_ylabel())
        ax.set_xlim(classifier.ax.get_xlim())
        ax.set_ylim(classifier.ax.get_ylim())

        for px, py, particle_class in zip(classifier.x_coords, classifier.y_coords, classifier.classes):
            color = COLOR_MAP.get(particle_class, 'black')
            ax.plot(px, py, '+', color=color, markersize=5)

        plt.draw()
        print(f"Removed particle at ({removed_x:.2f}, {removed_y:.2f}) of class {removed_class}.")
    else:
        print("No particle found within threshold for deletion.")


def on_keypress(event, ax, classifier, im, ppar):
    if event.key in ['1', '2', '3', '4']:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            particle_class = int(event.key)
            classifier.add_particle(x, y, particle_class)
            plt.draw()
    elif event.key == '5':
        classifier.save_particles(filename="pdParticles.pkl")
        df = classifier.get_coordinates()
        df.to_csv(f"{ppar.output_file}.txt", index=False)
        plt.savefig(f"{ppar.output_file}.png")
        print("All outputs saved successfully.")
    elif event.key == '6':
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            del_bkg_point_close_to_mouse(classifier, x, y, ax, im, threshold=10)
    elif event.key == 'q':
        on_close(None, ppar, classifier)
        plt.close()
        print("Plot closed.")


def on_close(event, ppar, classifier):
    classifier.save_particles(filename="pdParticles.pkl")
    df = classifier.get_coordinates()
    df.to_csv(f"{ppar.output_file}.txt", index=False)
    plt.savefig(f"{ppar.output_file}.png")
    print("Plot closed and data saved.")


# =============================================================================
# Level 3: Particle Classifier Class

class ParticleClassifier:
    def __init__(self, ax=None):
        self.x_coords = []
        self.y_coords = []
        self.classes = []
        self.notes = []
        self.output = pd.DataFrame()
        self.plot_points = []
        self.ax = ax if ax else plt.subplots()[1]

    def get_coordinates(self) -> pd.DataFrame:
        self.output = pd.DataFrame({
            "X": [round(x, 2) for x in self.x_coords],
            "Y": [round(y, 2) for y in self.y_coords],
            "Class": self.classes,
            "Note": self.notes,
        })
        return self.output

    def add_particle(self, x: float, y: float, particle_class: int) -> None:
        x, y = round(x, 2), round(y, 2)
        self.x_coords.append(x)
        self.y_coords.append(y)
        self.classes.append(particle_class)
        self.notes.append(CLASS_NOTES.get(particle_class, "Unknown"))

        color = COLOR_MAP.get(particle_class, 'black')
        plot_point, = self.ax.plot(x, y, '+', color=color, markersize=5)
        self.plot_points.append(plot_point)
        print(f"Added particle at (X={x}, Y={y}) as Class={particle_class}")

    def save_particles(self, filename: str = "pdParticles.pkl") -> None:
        self.get_coordinates().to_pickle(filename)
        print(f"Particles saved to '{filename}'.")


# =============================================================================
# Level 4: Aux Functions

def initialize_interactive_plot_parameters():
    plt.rcParams.update({
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'font.size': 12,
        'lines.linewidth': 1.0
    })


def clear_plot():
    ax = plt.gca()
    xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.cla()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)


def default_plot_params():
    class DefaultParams:
        xlabel = "X-axis"
        ylabel = "Y-axis"
        xlim = [0, 1000]
        ylim = [0, 1000]
        output_file = "output"
        pdParticles = "output"
    return DefaultParams()


# =============================================================================
# Example Usage

if __name__ == "__main__":
    class MockPlotParams:
        xlabel = "X-axis"
        ylabel = "Y-axis"
        xlim = [0, 1000]
        ylim = [0, 1000]
        output_file = "output"
        pdParticles = "output"
        messages = True

    ppar = MockPlotParams()
    fig, ax = ...  # Replace this with actual image loading and call to interactive_plot()
