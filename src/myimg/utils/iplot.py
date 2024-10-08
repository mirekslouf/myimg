'''
Module: bground.iplot
---------------------
The module with functions to create an interactive plot.

The interactive plot can be defined/created at three levels/steps:
    
* Level 1 = func yielding the interactive plot = the plot linked with events.
* Level 2 = funcs for event types (such as key_press_event, close_event, ...).
* Level 3 = funcs for individual sub-events (such as specific keypress events).

Important technical notes:

* We keep the (very reasonable) mouse events from Matplotlib UI.
* We define just *key_press_events*, with which we can do the whole job.
* We can define additional simple events (here: *close_event* = on closing).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bground import bdata, bfunc
import warnings; warnings.filterwarnings("ignore")

# =============================================================================
# Level 1: Create plot with events

def interactive_plot(data, bkgr, ppar):
    '''
    Create plot from input data.
    
    This is a plot window, which will be made interactive later.
    In the rest of the program, the plot will be the program interface.

    Parameters
    ----------
    data : 2D numpy array
        Data for plotting; columns [X,Y].
    bkgr : bground.bdata.XYbackground object
        An object storing data for the background calculation.
    ppar : bground.ui.PlotParameters object
        An object with data for interactive plot + name of output file
    
    Returns
    -------
    fig,ax : maptplotlib.pyplot objects
        The figure and axis of the interactive plot which shows XY-data.
        
    Technical notes
    ---------------
    The arguments/objects for this function (data, bkgr, ppar)
    are usually defined by means of a simple OO-interface
    before this function is called.
    '''
    
    # (0) Initialize
    plt.close('all')  # Close all previous plots - to avoid mess in Jupyter
    initialize_interactive_plot_parameters()
    
    # (1) Prepare the plot: fig,ax including window title
    # (num argument below can take both integers and strings
    fig,ax = plt.subplots(num='Background correction')

    # (2) Read XY data and create the plot
    # Get XY data from the function argument data
    X,Y = (data[0],data[1])
    # Plot XY data
    ax.plot(X,Y, 'b-')
    # Set the remaining plot parameters
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)
    ax.set_xlabel(ppar.xlabel)
    ax.set_ylabel(ppar.ylabel)
    
    # (3) Connect the plot with event(s)
    #   => link fig.canvas event(s) to a callback function(s).
    # Here: two types of events:
    #   key_press_event -> callback function = on_keypress.
    #   close_event     -> callback function = on_close
    # The callback function links the events to further user-defind funcs.
    #   Note1: all is based on standard matplotlib function canvas.mpl_connect
    #   Note2: the individual functions are defined below.
    # Trick: If we need an event with multiple arguments => lambda functions.
    #   GoogleSearch: python calback with multiple arguments 
    
    # (3a) Connect plot with key_press_event (= when a key is pressed)
    fig.canvas.mpl_connect('key_press_event',
        lambda event: on_keypress(event, fig, ax, data, bkgr, ppar))
    
    # (3b) Connect plot with close_event (= when the window is closed)
    fig.canvas.mpl_connect('close_event',
        lambda event: on_close(event, ppar))
    
    # (4) Optimize the plot layout
    plt.tight_layout()
    
    # (5) Return fig,ax
    # (This is necessary, among others, fof Jupyter + %matplotlib widget
    return(fig,ax)


# =============================================================================
# Level 2: Callback functions for all events

def on_keypress(event, fig, ax, data, bkgr, ppar):
    '''
    Definition of key_press_events for a plot.
    
    The callback function, which defines all keypress events.
    The functions for the individual keypress events are defined below.
    '''
    # Step 3 in defining interactive plot
    # = defining individual functions for specific pressed keys.
    # -----
    # Read pressed key and mouse coordinates
    key = event.key
    xm,ym = event.xdata,event.ydata
    # Mouse outside graph area - just print warning!
    if xm == None or ym == None:
        if ppar.messages:
            print(f'Key [{key}] mouse outside plot area - no action!')
    # Mouse inside graph area, run corresponding function.
    else:
        if ppar.messages:
            print(f'Key [{key:s}] mouse [{xm:.1f},{ym:.1f}]', end=' ')
        # Functions run by means try-except
        # Reason: to ignore nonsense actions...
        # ...such as delete/draw points if no points are defined
        try:
            if   key == '1': add_bkg_point(plt,data,bkgr,ppar,xm,ym)
            elif key == '2': del_bkg_point_close_to_mouse(plt,bkgr,ppar,xm,ym)
            elif key == '3': replot_with_bkg_points(plt,data,bkgr,ppar)
            elif key == '4': replot_with_bkg(plt,data,bkgr,ppar,'linear')
            elif key == '5': replot_with_bkg(plt,data,bkgr,ppar,'quadratic')
            elif key == '6': replot_with_bkg(plt,data,bkgr,ppar,'cubic')
            elif key == 'a': load_bkg_points(plt,data,bkgr,ppar)
            elif key == 'b': save_bkg_points(bkgr,ppar)
            elif key == 't': subtract_bkg_and_save(plt, data, bkgr, ppar)
            elif key == 's': save_PNG_image(ppar) 
            elif ppar.messages: print() # for any other key just empty line
        except Exception:
            pass


def on_close(event, ppar):
    '''
    Definition of on_close event of the plot.
    
    The simple callback function, which runs
    when the interactive plot window is closed.
    The function just prints some concluding remarks
    and information about the output files.
    '''
    out_file1 = ppar.output_file
    out_file2 = ppar.output_file + '.bkg'
    out_file3 = ppar.output_file + '.png'
    if not(out_file1.lower().endswith('.txt')): out_file1 = out_file1 + '.txt'
    print()
    print('The interactive plot was closed.')
    print('If you followed the instructions on www,')
    print('the outputs should be saved in the following files:')
    print(f' - {out_file1} = TXT file with background-corrected XYdata')
    print(f' - {out_file2} = BKG file containing background points')
    print(f' - {out_file3} = PNG plot of XY-data with defined background')


# =============================================================================
# Level 3: Functions for individual keypress events       

def add_bkg_point(plt, data, bkgr, ppar, xm, ym):
    '''
    Function for keypress = '1'.
    
    Add background point at current mouse position.
    More precisely: add a background point at the the XY-curve,
    whose X-coordinate is the closest to the mouse X-coordinate.
    '''
    idx = find_nearest(data[0],xm)
    xm,ym = (data[0,idx],data[1,idx])
    bkgr.points.add_point(xm,ym)
    plt.plot(xm,ym,'r+')
    plt.draw()
    if ppar.messages: print('background point added.')
    

def del_bkg_point_close_to_mouse(plt, bkgr, ppar, xm, ym):
    '''
    Function for keypress = '2'.
    
    Remove the background point close to mouse cursor position.
    More precisely: remove the background point from the XY-curve,
    whose X-coordinate is the closest to the mouse cursor X-coordinate.
    '''
    # a) Sort bkg points (sorted array is necessary for the next step)
    bfunc.sort_bkg_points(bkgr)
    # b) Find index of background point closest to the mouse X-position
    idx = find_nearest(np.array(bkgr.points.X), xm)
    # c) Remove element with given index from X,Y-lists (save coordinates)
    xr = bkgr.points.X.pop(idx)
    yr = bkgr.points.Y.pop(idx)
    # d) Redraw removed element with background color
    plt.plot(xr,yr, 'w+')
    # e) Redraw plot
    plt.draw()
    # f) Print message to stdout.
    if ppar.messages: print('background point deleted.')


def replot_with_bkg_points(plt, data, bkgr, ppar, points_reloaded=False):
    '''
    Function for keypress = '3'.
    
    Re-draw plot with backround points.
    Additional keyword parameter (points_reloaded=False),
    is employed if the function is called from load_bkg_points function.
    The load_bkg_points function uses its own short message
    and sets points_reloaded=True to avoid additional confusing messaging.
    '''
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkgr.points.X,bkgr.points.Y,'r+')
    plt.draw()
    if ppar.messages == True and points_reloaded == False:
        # 
        print('backround points re-drawn.')

def replot_with_bkg(plt, data, bkgr, ppar, btype):
    '''
    Function for keypress = '4,5,6'.
    
    Re-draw plot with backround points and background curve.
    Type of the curve is given by parameter btype.
    For key = 4/5/6 the function called with btype = linear/quadratic/cubic.
    '''
    # Sort background points + calculate background
    bfunc.sort_bkg_points(bkgr)
    bkgr.btype = btype
    bfunc.calculate_background(data, bkgr)
    # Clear plot + re-draw it with the background points
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkgr.points.X,bkgr.points.Y,'r+')
    plt.plot(bkgr.curve.X,bkgr.curve.Y,'r:')
    plt.draw()
    # Print a brief message on stdout if requested
    if ppar.messages:
        if bkgr.btype == 'linear':
            print('linear background displayed.')
        elif bkgr.btype == 'quadratic':
            print('quadratic background displayed.')
        elif bkgr.btype == 'cubic':
            print('cubic background displayed.')


def load_bkg_points(plt, data, bkgr, ppar):
    '''
    Function for keypress = 'a'.

    Load background points from previously saved file.
    Assumption: the file has been saved with save_bkg_points function,
    which means that its name is fixed and known (bkgr.basename + '.bkg').
    '''
    # a) get input file with previously saved background points
    # (the filename is fixed to [output_file_name].bkg
    # (reason: inserting a name during an interactive plot session is a hassle
    # (solution: manual renaming of the BKG-file before running this program
    input_filename = bkgr.basename + '.bkg'
    # b) read input file to DataFrame
    df = pd.read_csv(input_filename, sep=r'\s+')
    # c) initialize bkg object by means of above-read DataFrame
    bkgr.points = bdata.XYpoints(X = list(df.X), Y = list(df.Y))
    bkgr.btype='linear'
    # d) print message if requested
    if ppar.messages:
        print(f'background points read from: [{input_filename}].')
    # e) replot with currently loaded background
    replot_with_bkg_points(plt, data, bkgr, ppar, points_reloaded=True)


def save_bkg_points(bkgr, ppar):
    '''
    Function for keypress = 'b'.
    
    Save background points to file.
    The output file name is bkgr.basename + extension 'bkg'.
    '''
    # Treat the special case when the user presses 'b'
    # but there are no background points defined at the moment.
    if len(bkgr.points.X) == 0:
        print('no background points defined!')
        return()
    # Standard processing of event 'b' = saving of the background points.
    bfunc.sort_bkg_points(bkgr)
    output_filename = bkgr.basename + '.bkg'
    df = bkg_to_df(bkgr)
    with open(output_filename, 'w') as f:
        f.write(df.to_string())
        if ppar.messages:
            print(f'background points saved to: [{output_filename}]')
    

def subtract_bkg_and_save(plt, data, bkgr, ppar):
    '''
    Function for keypress 't'.
    
    This is the final function which:
    
    * Recalculates recently defined background
    * Calculates background-corrected data = subtracts bkg from data
    * Saves the results to TXT-file with 3 cols [X, Y, bkg-corrected-Y]
    '''
    # Subtract recently defined background and save results
    # (a) Recalculate background
    bfunc.calculate_background(data,bkgr)
    # (b) Subtract background
    data = bfunc.subtract_background(data,bkgr)
    # (c) Save background-corrected data to TXT-file
    # (we will use ppar object properties for this
    # (ppar.output_file = output file name, ppar.xlabel = label of X-data...
    output_filename = ppar.output_file
    if not(output_filename.lower().endswith('.txt')):
        output_filename = output_filename + '.txt'
    file_header = (
        f'Columns: {ppar.xlabel}, {ppar.ylabel}, ' +
        f'background-corrected-{ppar.ylabel}\n' +
        f'Background correction type: {bkgr.btype}')
    np.savetxt(
        output_filename, 
        np.transpose(data),
        fmt=('%8.3f','%11.3e','%11.3e'),
        header=file_header)
    if ppar.messages:
        print(f'backround-corrected data saved to: [{output_filename}]')


def save_PNG_image(ppar):
    '''
    Function for keypress 's'.
    
    Special case - 's' is Matplotlib UI shortcut,
    which saves PNG image of the current (interactive) plot.
    
    As key_press_event 's' was NOT disconnected from the plot,
    the following two things are going to happen:
    
    * At first, the default event (saving as PNG) will take place.
    * At second, this function prints a message on stdout (if requested).
    '''
    
    # 0) If 's' was pressed the current plot is saved automatically.
    # (Note: default Matplotlib UI shortcut
    
    # 1) We define the output filename.
    # (Note: this is only RECOMMENDED filename - user can select anything...
    output_filename = ppar.output_file + '.png'
    
    # 2) We print the message that the plot was saved
    # (Note: we add the info about the recommended filename
    if ppar.messages:
        print(f'plot saved to PNG; recommended name: [{output_filename}]')


# =============================================================================
# Level 4: Auxiliary functions for the interactive plot


def initialize_interactive_plot_parameters():
    '''
    Initialize parameters of the interactive plot.
    '''
    plt.rcParams.update({
        'figure.figsize' : (6,4),
        'figure.dpi' : 100,
        'font.size' : 12,
        'lines.linewidth' : 1.0})


def print_brief_help(ppar):
    '''
    Print ultra-brief help before activating the interactive plot.
    
    Parameters
    ----------
    ppar : 
        It is used just to get the value of ppar.messages.
        If ppar.messages == True, additional empty line is printed.
        Reason: To separate the introductory help from the following messages.
    
    Returns
    -------
    None
        The output is the brief help printed on stdout.
    '''
    print('(0) Click on the newly-opened window {Background correction},')
    print('    which is a Matplotlib interative figure with extra shortcuts.')
    print('(1) Use Matplotlib UI + keyboard shortcuts to define a background:')
    print('    1,2 = add/delete a background point close to the mouse cursor')
    print('    3,4,5,6 = show bkg points + linear/quadratic/cubic background')
    print('(2) Save the results + close the window when you are done:')
    print('    b = save the background points as a BKG-file (a = restore)')
    print('    t = subtract the background & save the result as a TXT-file')
    print('(3) Detailed help, complete documentation, and worked examples:')
    print('    https://mirekslouf.github.io/bground/docs')
    # Add an extra empty line if short messages to stdoud should be printed.
    if ppar.messages: print()


def find_nearest(arr, value):
    '''
    Auxiliary function:
    Find the index of the element with the nearest {value} in 1D-array.
    
    Parameters
    ----------
    arr : 1D numpy array
        The array, in which we search the element with closest value.
        Important prerequisite: the array must be sorted.
    value : float
        The value, for which we search the closest element.

    Returns
    -------
    idx : int
        Index of the element with the closest value.
    '''
    # Find index of the element with nearest value in 1D-array.
    # Important prerequisite: the array must be sorted.
    # https://stackoverflow.com/q/2566412
    # 1) Key step = np.searchsorted
    idx = np.searchsorted(arr, value, side="left")
    # 2) finalization = consider special cases and return final value
    if idx > 0 and (
            idx == len(arr) or abs(value-arr[idx-1]) < abs(value-arr[idx])):
        return(idx-1)
    else:
        return(idx)


def clear_plot():
    '''
    Auxilliary function: clear interactive plot before re-drawing.
    
    Key feature of the function:
    It keeps current labels and XY-limits of the plot.
    '''
    my_xlabel = plt.gca().get_xlabel()
    my_ylabel = plt.gca().get_ylabel()
    my_xlim = plt.xlim()
    my_ylim = plt.ylim()
    plt.cla()
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    plt.xlim(my_xlim)
    plt.ylim(my_ylim)
    

def bkg_to_df(bkgr):
    '''
    Auxiliary function: Convert current background points to dataframe.
    
    Why this function?
    => df can be used to save/restore background points nicely.
    '''
    # Convert bkg to DataFrame to get nicely formated output
    # (our trick: df.to_string & then print/save to file as string
    # (more straightforward: df.to_csv('something.txt', sep='\t')
    # (BUT the output with to_string has better-aligned columns
    df = pd.DataFrame(
        np.transpose([bkgr.points.X, bkgr.points.Y]), columns=['X','Y'])
    return(df)
