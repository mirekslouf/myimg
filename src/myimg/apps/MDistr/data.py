'''
Module: myimg.apps.MDistr.data
------------------------------

* Calculate correct size/shape distributions from multiple datafiles.
* The datafiles come from image analysis of multiple micrographs.
* The micrographs can have different magnifications.
'''

# General modules
import numpy as np
import pandas as pd

# Dataclasess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileInfo:
    filename: str
    Pmin: float
    Pmax: float
    file_weight: float


@dataclass
class PandasReadOptions:
    weights: bool = False
    sep: str = r"\s+"
    usecols: list[int] | list[str] | None = None
    additional: dict[str, Any] = field(default_factory=dict)


class Particles:
    
    # Wrapper for CombinedData.particles.
    # It contains df (DataFrame) + additional methods.
    # Key advantage: the additional methods enable chaining.
    # Key limitation: after chaining, we get DataFrame, not Particles object.
    # -------
    # >>> import myimg.api as mi
    # >>> MDistr = mi.Apps.import_MDistr()
    # >>> ...
    # >>> combined_data = CombinedData(files, read_options)
    # >>> combined_data.particles.df
    # >>> combined_data.particles.save('my_particles.txt')

    def __init__(self, df):
        self.df = df


    def show(self, my_round=None):
        
        df = self.df
        
        if isinstance(my_round, (int,np.integer)):
            # Convert to string, format all floats to {my_round} decimals 
            n = my_round
            fmt = f"{{:.{n}f}}".format
            print(df.to_string(float_format=fmt))
        else:
            # Natural pandas style, no rounding
            print(df)


    def save(self, path):
        self.df.to_csv(path, index=False)


class CombinedData:

    
    def __init__(self, files, read_options, name=None):
        
        # (a) Standard parameters
        #     needed to calculate the combinded_data
        self.files = files
        self.read_options = read_options
        
        # (b) Calculate DataFrame containing all particles with weights
        #     and store it as Particles object = DataFrame with methods
        df_with_combined_data = self.combine_multiple_files()
        self.particles = Particles(df_with_combined_data)
        
    
    def read_file(self, file):
        
        # (1) Read file using Pandas
        # (the file can have multiple columns = morphology descriptiors
        # (the 1st (obligatory) column = particle sizes
        # (the last (optional) column = particle weights/numbers
        df = pd.read_csv(
            file.filename,
            sep = self.read_options.sep,
            usecols = self.read_options.usecols,
            **self.read_options.additional)
        
        # (2) Process the first column = particle sizes
        # (set the column name to 'Size' if it is just default 0th column
        # (unnamend columns => 0,1,2... => rename the 1st col to 'Size'
        if df.columns[0] == 0:
            df = df.rename(columns={df.columns[0]: 'Size'})
        
        # (3) Process the last column = particle weights
        # (3a) Make sure that we have/create last column named 'Weight'
        if self.read_options.weights is True:
            # The last column should contain weights => rename it to 'Weight'
            df = df.rename(columns={df.columns[-1]: 'Weight'})
        else:
            # No last column with 'Weights' => create it, filled with ones
            df['Weight'] = 1
        # (3b) For each particle: weight = particle_weight * file_weight
        df['Weight'] = df['Weight'] * file.file_weight
        
        # (4) Select particles in the desired size range
        first_column = df.iloc[:,0]
        mask = (file.Pmin <= first_column) & (first_column < file.Pmax)
        df = df[mask]
        
        # (5) Return the processed file
        # (DataFrame with the following columns:
        # ( 1st column  => values: particle_size
        # ( any colunns => values: (optional) other morphological descriptors
        # ( last column => values: final_weight = particle_weight * file_weight
        return(df)
    
    
    def combine_multiple_files(self):
        
        list_of_dataframes = []
        
        for file in self.files:
            df = self.read_file(file)
            list_of_dataframes.append(df)
        
        df = pd.concat(list_of_dataframes, ignore_index=True)
    
        self.combined_measurements = df
    
        return(df)


class Statistics:

    
    def __init__(self, df: pd.DataFrame):
        
        # save input df (needed for all methods)
        self.data = df
        
        # Split and save features and weights
        self.values = self.data.iloc[:, :-1]
        self.weights = self.data.iloc[:, -1].astype(float)
        
        # Normalize weights
        # (good for numerical stability + simplification of formulas
        self.weights = self.weights/np.sum(self.weights)
        
        # calulate statistics
        self.stats = self.calculate()


    def calculate(self):
        
        # Prepare empty DataFeame for statistic results
        stats = pd.DataFrame(index=["Min", "Max", "Mean", "StDev"])
        
        # Prepare weights
        # (already normalized during initialization => sum(weights) = 1
        w = self.weights
        
        # Calculate the statistics
        for col in self.values.columns:
            x = self.values[col].astype(float)
            # weighted mean
            mean = np.sum(w * x)
            # weighted variance and stddev: sum w*(x-mean)^2 / sum w
            var = np.sum(w * (x - mean) ** 2)
            std = np.sqrt(var)
            # save the calclated values
            stats.loc["Min", col] = x.min()
            stats.loc["Max", col] = x.max()
            stats.loc["Mean", col] = mean
            stats.loc["StDev", col] = std

        # Return the result = DataFrame containing basic statistics
        # (basic statistical parameters for each column
        return stats


    def show(self, my_round=None):
        
        df = self.stats
        
        if isinstance(my_round, (int,np.integer)):
            # Convert to string, format all floats to {my_round} decimals 
            n = my_round
            fmt = f"{{:.{n}f}}".format
            print(df.to_string(float_format=fmt))
        else:
            # Natural pandas style, no rounding
            print(df)


    def save(self, filename: str):
        self.stats.to_csv(filename)


class Histogram:

    
    def __init__(self, df, column, bins):

        self.data = df
        self.bins = np.asarray(bins)

        # interpret column selection
        if isinstance(column, int):
            self.column_name = df.columns[column]
        else:
            self.column_name = column

        # core data
        self.x = df[self.column_name].astype(float)
        self.size = df.iloc[:, 0].astype(float)  # particle size (always col 0)
        self.w = df.iloc[:, -1].astype(float)    # weights (always last col)

        # compute distribution
        self.distribution = self.calculate()


    def calculate(self) -> pd.DataFrame:

        bins = self.bins
        n_bins = len(bins) - 1
        bin_width = bins[1]-bins[0]

        # output table
        out = pd.DataFrame(index=range(n_bins + 2),
            columns=["Min", "Max", "Center", "N", "V", "N%", "V%"],
            dtype=float)

        # bin indices for each particle
        idx = np.digitize(self.x, bins) - 1

        # extend with underflow/overflow bins
        under = idx < 0
        over = idx >= n_bins
        valid = ~(under | over)

        # bin accumulators
        N = np.zeros(n_bins)
        V = np.zeros(n_bins)

        # main bins
        np.add.at(N, idx[valid], self.w[valid])
        np.add.at(V, idx[valid], self.w[valid] * self.size[valid]**3)

        # underflow / overflow bins
        N_under = self.w[under].sum()
        V_under = (self.w[under] * self.size[under]**3).sum()

        N_over = self.w[over].sum()
        V_over = (self.w[over] * self.size[over]**3).sum()

        # total for normalization
        N_total = N_under + N.sum() + N_over
        V_total = V_under + V.sum() + V_over

        # fill table
        for i in range(n_bins + 2):
        
            if i == 0:
                label_min = -np.inf
                label_max = bins[0]
                center = bins[0] - bin_width/2
                n = N_under
                v = V_under
        
            elif i == n_bins + 1:
                label_min = bins[-1]
                label_max = np.inf
                center = bins[-1] + bin_width/2
                n = N_over
                v = V_over
        
            else:
                j = i - 1  # shift index for real bins
        
                label_min = bins[j]
                label_max = bins[j + 1]
                center = (bins[j] + bins[j+1]) / 2
                n = N[j]
                v = V[j]

            out.iloc[i, 0] = label_min
            out.iloc[i, 1] = label_max
            out.iloc[i, 2] = center
            out.iloc[i, 3] = n
            out.iloc[i, 4] = v
            out.iloc[i, 5] = 100 * n / N_total if N_total else 0
            out.iloc[i, 6] = 100 * v / V_total if V_total else 0

        return out


    def show(self, my_round=None):
        
        df = self.distribution
        
        if isinstance(my_round, (int,np.integer)):
            # Convert to string, format all floats to {my_round} decimals 
            n = my_round
            fmt = f"{{:.{n}f}}".format
            print(df.to_string(float_format=fmt))
        else:
            # Natural pandas style, no rounding
            print(df)


    def save(self, filename: str):
        self.distribution.to_csv(filename, index=False)
