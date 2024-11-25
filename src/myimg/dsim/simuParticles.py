# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 06:23:59 2024

@author: Pavlina Sikorova 
"""

import numpy as np
import matplotlib.pyplot as plt
from os import getcwd, sep
import pandas as pd
from PIL import Image


class dataGenerator():
    
    def __init__(self, imWidth=512, imHeight=0, numP=4, disp=True, 
                     smallParticle=(1, 5), bigParticle=(10, 15),
                     thrGauss=0.01, blurAmp=0.5):
        """
        Initialize the dataGenerator class with options for controlling
        particle sizes, sharpness, and display preferences.
        
        Parameters
        ----------
        imWidth : int, optional
            Image width (pixels). Default is 512.
        imHeight: int, optional
            Image height (pixels). Default is 0.
        numP : int, optional
            Number of particles to simulate. Default is 4.
        disp : bool, optional
            Whether to display the images. Default is True.
        smallParticle : tuple, optional
            Size range for small particles (min, max). Default is (1, 5).
        bigParticle : tuple, optional
            Size range for big particles (min, max). Default is (10, 15).
        thrGauss : float, optional
            Threshold for sharpness in Gaussian. Default is 0.01.
        blurAmp : float, optional
            Amplitude for blurry particles. Default is 0.5.
        """
        # Image size (dimension of the image)
        self.imWidth = imWidth   
        
        if imHeight==0:
            self.imHeight=imWidth
        else:
            self.imHeight=imHeight
        
        # Number of particles
        self.numP = numP
        
        # Store other parameters
        self.smallParticle = smallParticle
        self.bigParticle = bigParticle
        self.thrGauss = thrGauss
        self.blurAmp = blurAmp
        
        # Generate particle locations and simulate image
        self.im_simulator()
        
        # Prepare segments and classes
        self.segments = [1, 1, 1, 1]
        

        if numP==4:
            # Generate particle shapes
            self.generate_4G2D()
            # Store coordinates and final image
            self.get_coordinates()
            self.get_simulation(show=disp)
        else:
            self.generate_moreG2D()
            # Store coordinates and final image
            self.get_coordinates()
            self.get_simulation(show=disp)


    def im_simulator(self):
        '''
        Simulate an image with random points.

        Returns
        -------
        None.

        '''
        # Initialize background image with zeros
        self.image = np.zeros((self.imWidth, self.imHeight)) 
        
        # Random x coordinates
        self.x_coords = np.random.randint(30, self.imWidth-30, size=self.numP) 
        
        # Random y coordinates
        self.y_coords = np.random.randint(30, self.imHeight-30, size=self.numP)  
        

    def gaussian2D(self, X, Y, h, x0, y0, sx, sy):
        '''
        Create a 2D Gaussian distribution.
        
        Parameters
        ----------
        X : np.ndarray
            A meshgrid of x-coordinates.
        Y : np.ndarray
            A meshgrid of y-coordinates.
        h : float
            The amplitude (maximum height) of the Gaussian distribution.
        x0 : float
            The x-coordinate of the center of the Gaussian.
        y0 : float
            The y-coordinate of the center of the Gaussian.
        sx : float
            The standard deviation of the Gaussian along the x-axis, 
            controlling its width in the x direction.
        sy : float
            The standard deviation of the Gaussian along the y-axis, 
            controlling its width in the y direction.
    
        Returns
        -------
        np.ndarray
            A 2D array representing the Gaussian distribution over 
            the specified grid.
        '''
        return h*np.exp(-((X-x0)**2/(2*sx**2)+(Y-y0)**2/(2*sy**2)))


    def gaussian2D_sharp(self, X, Y, h, x0, y0, sx, sy, threshold=0.7):
        '''
        Create a 2D Gaussian distribution with sharp edges by applying 
        a threshold.
    
        Parameters
        ----------
        X : np.ndarray
            A meshgrid of x-coordinates.
        Y : np.ndarray
            A meshgrid of y-coordinates.
        h : float
            The amplitude (maximum height) of the Gaussian distribution.
        x0 : float
            The x-coordinate of the center of the Gaussian.
        y0 : float
            The y-coordinate of the center of the Gaussian.
        sx : float
            The standard deviation of the Gaussian along the x-axis, 
            controlling its width in the x direction.
        sy : float
            The standard deviation of the Gaussian along the y-axis, 
            controlling its width in the y direction.
        threshold : float, optional
            A threshold value to create sharp edges. All values below this 
            threshold are set to zero. Default is 0.7.
    
        Returns
        -------
        gauss : np.ndarray
            A 2D array representing the Gaussian distribution with values 
            below the threshold set to zero, giving it sharper edges.
        '''
        # Gaussian calculation
        gauss = h*np.exp(-((X-x0)**2/(2*sx**2)+(Y-y0)**2/(2*sy**2)))
        
        # Apply threshold for sharpness
        gauss[gauss < threshold] = 0  
        return gauss


    def gauss_specs(self, grid_size=50,range2=(1,5),center=(0,0),amp=1, iso=1):
        '''
        Generate a grid and Gaussian distribution parameters with random 
        scaling for standard deviations.
    
        Parameters
        ----------
        grid_size : int, optional
            The size of the grid on which the Gaussian distribution will be 
            defined. It defines the range of the coordinates. Default is 50.
        range2 : tuple, optional
            A tuple defining the range (min, max) for randomly sampling the 
            standard deviation (sx, sy) of the Gaussian distribution. 
            Default is (1, 5).
        center : tuple, optional
            A tuple containing the x and y coordinates of the center of the 
            Gaussian distribution. Default is (0, 0).
        amp : float, optional
            The amplitude (maximum height) of the Gaussian distribution. 
            Default is 1.
    
        Returns
        -------
        X : np.ndarray
            A meshgrid of x-coordinates for the 2D Gaussian.
        Y : np.ndarray
            A meshgrid of y-coordinates for the 2D Gaussian.
        h : float
            The amplitude (maximum height) of the Gaussian distribution.
        x0 : float
            The x-coordinate of the center of the Gaussian.
        y0 : float
            The y-coordinate of the center of the Gaussian.
        sx : float
            The standard deviation of the Gaussian along the x-axis,
            randomly selected from the given range.
        sy : float
            The standard deviation of the Gaussian along the y-axis, 
            equal to sx for an isotropic Gaussian.
        '''
        
        # Generate random standard deviation in x and y directions
        sx = np.random.uniform(range2[0], range2[1])
        
        if iso:
            # isotropic gaussian
            sy = np.copy(sx)
        else: 
            # anisotropic gaussian
            sy= np.random.uniform(range2[0], range2[1])
            
        # A meshgrid of x- and y-coordinates for the 2D Gaussian
        x = np.linspace(0, grid_size, self.imWidth)
        y = np.linspace(0, grid_size, self.imHeight)
        X, Y = np.meshgrid(x, y)
        
        # Amplitude (maximum height of the distribution)
        h = amp
        
        # Coordinates of the center of the Gaussian
        x0, y0 = center[0], center[1]
        
        return X, Y, h, x0, y0, sx, sy


    def generate_4G2D(self):
        '''
        Generate all 2D Gaussian shapes based on different size and sharpness 
        scenarios, for a set number of particles.
    
        This method generates four different types of particles, each defined 
        by specific parameters for the 2D Gaussian or sharp Gaussian 
        (with a threshold):
        
        - Case 1: Small sharp circle (particleSS)    - Very sharp, small size
        - Case 2: Small blurry Gaussian (particleSB) - Blurry, small size
        - Case 3: Big sharp circle (particleBS)      - Very sharp, large size
        - Case 4: Big blurry Gaussian (particleBB)   - Blurry, large size
    
        The particles are positioned based on previously generated random 
        coordinates (`self.x_coords` and `self.y_coords`).
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
            This function generates + stores the particles as attributes:
            - `self.particleSS`: Small sharp circle
            - `self.particleSB`: Small blurry Gaussian
            - `self.particleBS`: Big sharp circle
            - `self.particleBB`: Big blurry Gaussian
        '''
        # Case 1: Small sharp circle (Very sharp, small size)
        X, Y, h, x0, y0, sx, sy = self.gauss_specs(
            grid_size=self.imWidth, 
            range2=self.smallParticle,
            center=(self.x_coords[0], self.y_coords[0]),
            amp=1
        )
        self.particleSS = self.gaussian2D_sharp(X, Y, h, x0, y0, sx, sy, 
                                                threshold=self.thrGauss)
    
        # Case 2: Small blurry Gaussian (Blurry, small size)
        X, Y, h, x0, y0, sx, sy = self.gauss_specs(
            grid_size=self.imWidth, 
            range2=self.smallParticle, 
            center=(self.x_coords[1], self.y_coords[1]),
            amp=self.blurAmp
        )
        self.particleSB = self.gaussian2D(X, Y, h, x0, y0, sx, sy)
    
        # Case 3: Big sharp circle (Very sharp, large size)
        X, Y, h, x0, y0, sx, sy = self.gauss_specs(
            grid_size=self.imWidth, 
            range2=self.bigParticle,  
            center=(self.x_coords[2], self.y_coords[2]),
            amp=1
        )
        self.particleBS = self.gaussian2D_sharp(X, Y, h, x0, y0, sx, sy, 
                                                threshold=self.thrGauss)
    
        # Case 4: Big blurry Gaussian (Blurry, large size)
        X, Y, h, x0, y0, sx, sy = self.gauss_specs(
            grid_size=self.imWidth, 
            range2=self.bigParticle, 
            center=(self.x_coords[3], self.y_coords[3]),
            amp=self.blurAmp
        )
        self.particleBB = self.gaussian2D(X, Y, h, x0, y0, sx, sy)

    
    def generate_moreG2D(self):
        self.particleSS, self.particleSB, self.particleBS, self.particleBB = \
            [], [], [], []
    
        # Calculate the integer number of iterations 
        iters = self.random_divide(num=self.numP)
        self.segments=np.copy(iters)
    
        xy = 0
        # Check that we have the right number of segments
        if len(iters) != 4:
            raise 
            ValueError("Expected 4 segments but got: {}".format(len(iters)))
    
        # Case 1: Small sharp Gaussian (Very sharp, small size)
        for _ in range(iters[0]):  
            X, Y, h, x0, y0, sx, sy = self.gauss_specs(
                grid_size=self.imWidth, 
                range2=self.smallParticle,
                center=(self.x_coords[xy], self.y_coords[xy]),
                amp=1
            )   
            
            self.particleSS.append(
                self.gaussian2D_sharp(X, Y, h, x0, y0, sx, sy, 
                                      threshold=self.thrGauss))
            xy=xy+1
          
        # Case 2: Small blurry Gaussian (Blurry, small size)
        for _ in range(iters[1]):  
            X, Y, h, x0, y0, sx, sy = self.gauss_specs(
                grid_size=self.imWidth, 
                range2=self.smallParticle, 
                center=(self.x_coords[xy], self.y_coords[xy]),
                amp=self.blurAmp
            )
            
            self.particleSB.append(
                self.gaussian2D(X, Y, h, x0, y0, sx, sy))
            xy=xy+1

    
        # Case 3: Big sharp Gaussian (Very sharp, large size)
        for _ in range(iters[2]):  # Use iters[2] for the number of particles
            X, Y, h, x0, y0, sx, sy = self.gauss_specs(
                grid_size=self.imWidth, 
                range2=self.bigParticle,  
                center=(self.x_coords[xy], self.y_coords[xy]),
                amp=1
            )
            self.particleBS.append(
                 self.gaussian2D_sharp(X, Y, h, x0, y0, sx, sy, 
                                                    threshold=self.thrGauss))
            xy=xy+1

        # Case 4: Big blurry Gaussian (Blurry, large size)
        for _ in range(iters[3]):  # Use iters[3] for the number of particles
            X, Y, h, x0, y0, sx, sy = self.gauss_specs(
                grid_size=self.imWidth, 
                range2=self.bigParticle, 
                center=(self.x_coords[xy], self.y_coords[xy]),
                amp=self.blurAmp
            )
            self.particleBB.append(
                 self.gaussian2D(X, Y, h, x0, y0, sx, sy))
    
            xy=xy+1
        
        
    def random_divide(self, num, parts=4):
        # Generate 3 random cut points between 0 and num
        cut_points = np.random.choice(range(1, num), 
                                      size=parts-1, 
                                      replace=False)
        
        # Sort the cut points
        cut_points.sort()
        
        # Calculate the lengths of the segments
        # The first segment from 0 to the first cut point
        segments = [cut_points[0]]  
        for i in range(1, len(cut_points)):
            # The segment between two cut points
            segments.append(cut_points[i] - cut_points[i-1]) 
        
        # The last segment from the last cut point to num
        segments.append(num - cut_points[-1])  
        
        return segments
    
    
    def get_coordinates(self):
        '''
        Store the particle coordinates and their corresponding classes in 
        a pandas DataFrame.
        
        This method creates a pandas DataFrame that stores the x, y coordinates 
        of the particles, as well as a class label for each particle. 
        The classes are assigned based on the segments defined in self.segments.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            The output DataFrame is stored in the instance variable self.output
            It contains the columns:
            - 'X': x-coordinates of particles
            - 'Y': y-coordinates of particles
            - 'Class': particle class label (1, 2, 3, 4)
            - 'Note' : nanoparticle specification for given class
        '''
        
        # Create lists to store data
        x_coords = []
        y_coords = []
        classes = []
        notes = []
        
        # Class specifications
        class_labels = [1, 2, 3, 4]
        class_notes = ["SS : Small Sharp", 
                       "SB : Small Blurry", 
                       "BS : Big Sharp", 
                       "BB : Big Blurry"]
    
        # Start index for accessing coordinates
        index = 0
    
        # Iterate over segments and populate the lists
        for class_index, segment in enumerate(self.segments):
            for i in range(segment):
                x_coords.append(self.x_coords[index])  # Access by index
                y_coords.append(self.y_coords[index])  # Access by index
                classes.append(class_labels[class_index])
                notes.append(class_notes[class_index])
                index += 1  # Increment the index
        
        # Create the DataFrame
        self.output = pd.DataFrame({
            "X": x_coords,
            "Y": y_coords,
            "Class": classes,
            "Note": notes
        })

    
    def get_simulation(self, show=1):
        '''
        Combine all generated particle images into a final image and display it 
        if required.
        
        Parameters
        ----------
        show : int, optional
            A flag to control whether the final image and 1D profile are 
            displayed. Default is 1.
        
        Returns
        -------
        imNanop : ndarray
            A 2D numpy array representing the combined image of all particles.
        '''
        # Initialize the final image with zeros
        self.imNanop = np.zeros((self.imWidth, self.imWidth))  
    
        # Ensure each particle variable is a list
        self.particleSS = self.particleSS \
            if isinstance(self.particleSS, list) else [self.particleSS]
        self.particleSB = self.particleSB \
            if isinstance(self.particleSB, list) else [self.particleSB]
        self.particleBS = self.particleBS \
            if isinstance(self.particleBS, list) else [self.particleBS]
        self.particleBB = self.particleBB \
            if isinstance(self.particleBB, list) else [self.particleBB]

        # Sum all generated nanoparticles to form the final image
        self.allparticles = (
            self.particleSS+self.particleSB+self.particleBS+self.particleBB)
        
        for particle in self.allparticles:
            self.imNanop += particle
    
        # Normalize the image values to 0-255
        min_val = self.imNanop.min()
        max_val = self.imNanop.max()
        
        # Avoid division by zero if all values are the same
        if max_val - min_val != 0:
            # Normalize to 0-1
            self.imNanop = (self.imNanop - min_val) / (max_val - min_val)  
            # Scale to 0-255 and convert to uint8
            self.imNanop = (self.imNanop * 255).astype(np.uint8)  
        else:
            # If all values are the same, make it a black image
            self.imNanop = np.zeros_like(self.imNanop, dtype=np.uint8)  


        # Show simulations
        if show:
            plt.figure()
            plt.imshow(self.imNanop, origin="lower", cmap="inferno")
            plt.title("Simulated Nanoparticles")
            plt.xticks([]); plt.yticks([])
            plt.tight_layout()
            plt.show()
    
            self.show1D(self.imNanop)
    
        return self.imNanop


    
    def show1D(self, im):
        '''
        Displays a 1D profile of the simulated nanoparticle image along with 
        class-specific markers.
        
        This function computes the 1D profile of the input image by summing
        pixel values across one axis (vertical profile). It then scales the 
        profile to the range [0, 1] and visualizes it. Additionally, vertical 
        lines are drawn at specific x-coordinates to represent the position
        of different classes of nanoparticles. The colors of the vertical lines 
        correspond to different particle classes.
        
        Parameters
        ----------
        im : ndarray
            A 2D numpy array representing the simulated image of nanoparticles.
    
        Returns
        -------
        None.
            Displays a plot of the 1D profile of the nanoparticle simulation 
            and marks particle positions based on their class.
        '''
        
        plt.figure()
        
        # Plot the 1D profile (sum across axis 0 for vertical profile)
        profile = np.sum(im, 0)
        
        # Scale the profile to the range 0 to 1
        profile_min = np.min(profile)
        profile_max = np.max(profile)
        
        if profile_max - profile_min > 0:  # Avoid division by zero
            profile = (profile - profile_min) / (profile_max - profile_min)
        else:
            profile = np.zeros_like(profile)  # If all values are the same
    
        plt.plot(profile, label="Nanoparticle profile")  
    
        # Define colors for each class (using integers as keys)
        class_colors = {
            1: "blue",   # Class 0: Small Sharp (SS)
            2: "orange", # Class 1: Small Blurry (SB)
            3: "green",  # Class 2: Big Sharp (BS)
            4: "red"     # Class 3: Big Blurry (BB)
        }
    
        # Track added classes to avoid duplicate legend entries
        added_classes = set()
    
        # Add vertical lines for each particle
        for i, xc in enumerate(self.x_coords):
            # Determine the class index based on the particle index
            class_idx = self.output["Class"][i]

            # Draw a vertical line at the x coordinate
            if i<20:
                plt.axvline(x=xc,color=class_colors[class_idx],linestyle='--')
    
            # Add the class label to the legend if not already added
            if class_idx not in added_classes:
                plt.plot([],[], color=class_colors[class_idx], linestyle='--', 
                         label=f'Class {class_idx}')  
                added_classes.add(class_idx)
        
        plt.xlabel("Image Width [Pixel] ")
        plt.ylabel("Normalized Intensity Sum [-]")

        plt.title("Profile of Simulated Nanoparticles")
        plt.legend(loc="best")
        plt.show()




if __name__ == "__main__":
    plt.close("all")
    
    for i in range(10):
        # Create a new instance of the dataGenerator class
        numP = int(np.round(np.random.uniform(4, 20)))
        dataGen = dataGenerator(imWidth=1024, numP=numP, disp=False)  
        
        # Get the simulated image and particle data
        imSimu = dataGen.imNanop
        data = dataGen.output
        
        # Save simulated data    
        path = getcwd() + sep  
        dfFile=f"pdParticles_{i}.pkl"
        data.to_pickle(path+dfFile)
        
        # Save the image
        imFile=f"imParticles_{i}.png"
        Image.fromarray(imSimu).save(path+imFile)
        
        # # Load data
        # data_loaded = pd.read_pickle(path+file)
        # im_loaded = np.array(Image.open(path+imFile))
    
        # plt.figure()
        # plt.imshow(im_loaded, cmap='inferno',  origin="lower")
        # plt.title('Loaded Image of Simulated particles')
        # plt.axis('off')  # Hide axis ticks
        # plt.show()
