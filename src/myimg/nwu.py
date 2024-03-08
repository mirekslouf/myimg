'''
Module: myimg.nwu
-----------------

This module contains three classes,
which enable work with numbers-with-units:

* Units class (and in its subclasses)
  describe allowed units.
* NumberWithUnits class defines
  NumberWithUnits object = number + units.
* ScaleWithUnits class defines
  ScaleWithUnits object = NumberWithUnits + length-in-pixels.

Typically, the three classes are used in myimg.utils.scalebar module
when creating scalebars (as a scalebar contains *number with units*).

Nevertheless, NumberWithUnits class can be used standalone, such as:
    
>>> # Simple usage of NumberWithUnits class
>>> from myimg.nwu import NumberWithUnits
>>> 
>>> n =NumberWithUnits('0.1mm')
>>> print('Initial number_with_units:', n)  # prints 0.1 mm
>>> 
>>> print('Setting correct units...')
>>> n.set_correct_units()
>>> print(n)                                # prints 100 um
'''


import sys, re
from dataclasses import dataclass

@dataclass
class Units:
    '''
    Data class: just a container for the following two dataclasses.
    
    * The sub-dataclasses are used in NumberWithUnits/ScaleWithUnits objects.
    * The units definitions are fixed, the users should not change them.
    * The correct units (Lenghts or RecLenghts) are guessed
      during NumberWithUnits/ScaleWithUnits object initialization.
      Example: If `nwu = NumberWithUnits('2um')`, then we use `Units.Lenghts`.
    '''
    
    @dataclass
    class Lengths:
        '''
        Data class: length units (and their ratios) for micrographs.    
        '''
        micro      : str = '\u00B5'
        angstrem   : str = '\u212B'
        micrometer : str = micro + 'm'
        units      : tuple = ('m','cm','mm', 'um', 'nm', 'A')
        ratios     : tuple = ( 1, 100, 1000, 1e6, 1e9, 1e10)
    
    @dataclass
    class RecLengths:
        '''
        Data class: reciprocal lenght units (and their ratios).
        
        * Reminder: reciprocal lenghts are used in diffractograms
        * Here: rust a pair of units that are employed in real life: 1/nm, 1/A
        '''
        angstrem      : str = '\u212B'
        rec_angstrem  : str =  '1/'+angstrem
        units         : tuple = ('1/nm','1/A')
        ratios        : tuple = ( 1, 10)


class NumberWithUnits:
    '''
    Class defining NumberWithUnits object.
    
    * NumberWithUnits object = number + units.
    * The numbers-with-units are used for pixel sizes or scalebars.
      
    Object initialiation
    
    >>> # Three basic ways how to initialize a NumberWithUnits object
    >>> from myimg.nwu import NumberWithUnits
    >>> nwu1 = ScaleWithUnits('1.2um')
    >>> nwu2 = ScaleWithUnits(number=1.2, units='um')
    >>> nwu3 = ScaleWithUnits(number_with_units_object_such_as_nwu1)
    
    List of object properties
        
    * number = number/numeric value
    * units = units corresponding to number
    * _units_description = myimg.settings.Units subclass, private property
    
    Object methods
    
    * text = return number-with-units as string
    * units_Ok = test if the defined units are correct
    * index_of_units = index of units in units_description.units list
    * increase_units = increase units and modify number accordingly
    * decrease_units = decrease units and modify number accordingly
    * set_units_to = set units to given units and modify number accordingly
    * set_correct_units = set units so that the number was within <1,1000)
    '''
    
    def __init__(self, number=None, units=None):
        '''
        Initialize NumberWithUnits object.

        Parameters
        ----------
        number : float or str or NumberWithUnits object
            Number with (or without) units, which can be:
            (i) Number (float; such as: 100),
            but then the 2nd argument should be given (such as units='um'). 
            (ii) String (str; such as: '100 um' or '1.2nm').
            (iii) Another NumberWithUnits object;
            in such a case we receive the copy of the argument.
              
        units : str, optional, the default is None
            If the 1st argument (number) is a float, the 2nd argument (units)
            defines units of the first argument.
            If the 1st argument is string or NumberWithUnits object,
            the 2nd argument is ignored.
            
        Returns
        -------
        NumberWithUnits object
            NumberWithUnits object contains:
            (i) numerical value (NumberWithUnits.number),
            (ii) corresponding units (NumberWithUnits.units), and
            (iii) further props/methods (NumberWithUnits.change_units ...).
        '''
        # (1) Determine input types
        if number is not None: input1 = type(number)
        if units  is not None: input2 = type(units)
        # (2) Parse the input number with units
        if input1 == float or input1 == int:
            # 1st input is a float (number) => 2nd input should be str (units)
            if input2 == str:
                self.number = number
                self.units  = units
            else:
                print('Error in defining NumberWithUnits object!')
                print('Units not given, exiting...')
                sys.exit()
        elif input1 == str:
            # input is string => split using regexp
            # TODO: consider errors and exceptions
            m = re.search(pattern='''
                # 1st/2nd group = number/units in standard () => to remember
                # exponent/rec.value in (?:) => not to remember
                (                  # start of 1st group = number
                  \d*\.?\d*        # number, such as 1, 1.2, .9
                  (?:e[+-]?\d+)?   # optional exponent, such as e3 e-6
                )                  # end of 1st group
                \s*                # possible whitespace
                (                  # start of 2nd group = units
                  (?:1/)?          # optional reciprocal value = '1/'
                  \D+              # one or more non-digit characters
                )                  # end of 2nd group
                ''',
                string = number,
                flags = re.VERBOSE)
            number,units = m.groups()
            self.number = float(number)
            self.units  = units
        elif input1 == NumberWithUnits:
            # input is another NumberWithUnits => create copy of it
            self.number = number.number
            self.units  = number.units
        else:
            print('Error in defining NumberWithUnits:')
            print('Wrong input number_with_units!')
            sys.exit()
        # (3) Determine, which units we use
        # => set private _units_description property
        if self.units in Units.Lengths.units:
            self._units_description = Units.Lengths
        elif self.units in Units.RecLengths.units:
            self._units_description = Units.RecLengths
        else:
            print('Error in defining NumberWithUnits:')
            print('Uknown units!')
            sys.exit()
    
    def __str__(self):
        '''Overwritten __str__ method to get nice output with print.'''
        # Get number and units (they will be adjusted, originals unchanged).
        number = self.number
        units  = self.units
        # Adjust number: if it is (very close to) integer, convert to integer.
        eps = 1e-10
        if abs(number - int(number)) < eps: number = int(number)
        # Adjust units: if units contains a special character, print it.
        if   units == 'um' : units = self._units_description.micrometer
        elif units == 'A'  : units = self._units_description.angstrem
        elif units == '1/A': units = self._units_description.rec_angstrem
        # Combine final number and units
        text = str(number) + ' ' + units
        # Return resulting string
        return(text)
    
    def text(self):
        '''
        Return number + units as string.

        Returns
        -------
        str : number-with-units
            The method returns the saved number with units as string.
            Example: if self.number = 1.2 and self.units = nm, we get '1 nm'.
            The units are printed in unicode (important for um and agstrems).
        '''
        return(self.__str__())
    
    def units_Ok(self, units=None):
        '''
        Test if current units are correct.

        Parameters
        ----------
        units : str, optional, the default is None
            If units='something', then 'something' is compared
            with the list of allowed units for self = NumberWithUnits object.

        Returns
        -------
        bool
            True if the units are correct and False otherwise.
        '''
        # (1) If units argument is not given,
        # check units of current NumberWithUnits object.
        if units is None:
            units = self.units
        # (2) Return True/False if units are correct/incorrect.
        if units in self._units_description.units:
            return(True)
        else:
            return(False)
    
    
    def index_of_units(self, units_to_check=None):
        '''
        Get index of current or specified units.

        Parameters
        ----------
        units_to_check : str, optional, default is None
            Any string representing some units.
            If units_to_check is not given,
            the method check current units of self object.
            
        Returns
        -------
        int
            Index of current or specified units.
            If units = 'um' then index_of_units = 2,
            because Units.Lenghts.units = ('m', 'mm', 'um', 'nm', 'A'),
            which means: 'm' => 0, 'mm' => 1, 'um' => 2...
        
        Note
        ----
        This function is employed in further functions
        manipulating with units, such as increase_units, decrease_units...
        '''
        if units_to_check == None: units_to_check = self.units 
        index_of_units = self._units_description.units.index(units_to_check)
        return(index_of_units)
    
    
    def increase_units(self):
        '''
        Increase current units (for example: um -> mm).

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        # (1) Get index of current units.
        i = self.index_of_units()
        # (2) If index-of-curent-units = 0,
        # then it is not possible to increase units - we are at maximum. 
        if i == 0:
            print('It is not possible to increase units!')
        # (3) If index-of-current-units > 0,
        # we can: 1) calculate ratio, 2) decrease number, 3) increase units.
        else:
            ratio_between_units = (
                self._units_description.ratios[i] /
                self._units_description.ratios[i-1] )
            self.number = self.number / ratio_between_units
            self.units  = self._units_description.units[i-1]


    def decrease_units(self):
        '''
        Decrease current units (for example: um -> nm).

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        # (1) Get index of current units.
        i = self.index_of_units()
        max_index = len(self._units_description.units) - 1
        # (2) If index-of-curent-units = last-index-of-our-units,
        # then it is not possible to decrease units - we are at minimum. 
        if i == max_index:
            print('It is not possible to decrease units!')
        # (3) If index-of-current-units < last-index-of-our-units,
        # we can: 1) calculate ratio, 2) increase number, 3) decrease units.
        else:
            ratio_between_units = (
                self._units_description.ratios[i] /
                self._units_description.ratios[i+1])
            self.number = self.number / ratio_between_units
            self.units  = self._units_description.units[i+1]
    
    
    def set_units_to(self, target_units):
        '''
        Set units to *target_units* and modify number accordingly.

        Parameters
        ----------
        target_units : str
            Any string specifying valid units.

        Returns
        -------
        None
            The results = changed units (and correspondingly changed number)
            are saved in NumberWithUnits object.
        '''
        index_current    = self.index_of_units()
        index_target     = self.index_of_units(target_units)
        index_difference = index_target - index_current
        if index_difference > 0:
            for i in range(index_difference): self.decrease_units()
        elif index_difference < 0:
            for i in range(abs(index_difference)): self.increase_units()
        
        
    def set_correct_units(self):
        '''
        Set correct units, so that the number was between 1 and 1000.

        Returns
        -------
        None
            The result is saved in NumberWithUnits object.
        '''
        while self.number < 1:
            self.decrease_units()
        while self.number >= 1000:
            self.increase_units()


class ScaleWithUnits(NumberWithUnits):
    '''
    Class defining ScaleWithUnits object.
    
    * ScaleWithUnits object = number + units + pixels.
    * The objet defines a scalebar:
      its lenght (number,units) and length-in-pixels (pixels).
    
    Object initialiation
    
    >>> # Three basic ways how to initialize a ScaleWithUnits object
    >>> from myimg.nwu import ScaleWithUnits
    >>> swu1 = ScaleWithUnits('1.2um', pixels=100)
    >>> swu2 = ScaleWithUnits(number=1.2, units='um', pixels=100)
    >>> swu3 = ScaleWithUnits(number_with_units_object, pixels=100)
    
    List of object properties
        
    * number = number/numeric value
    * units  = units corresponding to number
    * pixels = pixels corresponding to number-with-units
    * _units_description: myimg.settings.Units subclass, private property
    
    Object methods
    
    * inherited methods from myimg.nwu.NumberWithUnits
    * adjust_lenght_to = adjust lenght in pixels and modify number accordingly
    * adjust_scalebar_size = adjusts scalebar lenght to some reasonable size
    '''
    
    def __init__(self, number=None, units=None, pixels=None):
        '''
        Initialize ScaleWithUnits object.

        Parameters
        ----------
        number : float or str or NumberWithUnits object
            Number with (or without) units, which can be:
            (i) Number (float; such as: 100),
            but then the 2nd argument should be given (such as units='um'). 
            (ii) String (str; such as: '100 um' or '1.2nm').
            (iii) Another NumberWithUnits object;
            in such a case we receive the copy of the argument.
        units : str, optional, the default is None
            If the 1st argument (number) is a float, the 2nd argument (units)
            defines units of the first argument.
            If the 1st argument is string or NumberWithUnits object,
            the 2nd argument is ignored.
        pixels : float
            Length of scalebar in pixels.
            This is a keyword argument that is formally optional, but in fact 
            it must be specified so that the initialization made sense.
            Moreover, this argument must be specified as keyword argument,
            because the 2nd argument (units) is really optional and
            it may not be clear, which argument is which.
            
        Returns
        -------
        ScaleWithUnits object
            NumberWithUnits object contains:
            (i) numerical value (NumberWithUnits.number),
            (ii) corresponding units (NumberWithUnits.units),
            (iii) length-of-scalebar-in-pixels and
            (iv) further props/methods (most of which
            are inherited from NumberWithUnits superclass).
        '''
        super().__init__(number, units)
        self.pixels = pixels
    
    
    def __str__(self):
        '''Overwritten __str__ method to get nice output with print.'''
        text = \
            'Scalebar: ' + super().__str__() + \
            ', length-in-pixels: ' + str(self.pixels)
        return(text)
    
    def text(self):
        '''
        Return number-with-units as string (ignoring pixels).

        Returns
        -------
        str : number-with-units
            The method returns the saved number with units as string.
            Example: if self.number = 1.2 and self.units = nm, we get '1 nm'.
            The units are printed in unicode (important for um and agstrems).

        '''
        # We need JUST text.
        # => therefore, we call super().__str__() - this returns just text.
        # => if we called self.__str__(), we would get text + pixel-size-info.
        text = super().__str__()
        return(text)
    
    
    def adjust_length_to(self, n):
        '''
        Set lenght-of-scalebar to *n* and modify lenght-in-pixels accordingly.

        Parameters
        ----------
        n : float
            The new length-of-scalebar (= self.number).

        Returns
        -------
        None
            The new lenght-of-scalebar and lenght-of-scalebar-in-pixels
            are saved in ScaleWithUnits.number and ScaleWithUnits.pixels
            properties, respectively.
        '''
        self.pixels = self.pixels * n/self.number
        self.number = n
        
    
    def adjust_scalebar_size(self):
        '''
        Set scalebar to some reasonable lenght
        and modify lenght-in-pixels accordingly.

        Returns
        -------
        None
            The modified (number, units, and pixels) are saved in
            ScaleWithUnits object.
            
        Examples
        --------
        * swu = 0.3 um => swu.adjust_scalebar_size() => 300 nm
        * swu = 2.3 um => swu.adjust_scalebar_size() => 2 um
        * swu = 456 um => swu.adjust_scalebar_size() => 500 um
        * swu = 888 um => swu.adjust_scalebar_size() => 1 mm
        '''
        # Get the number in the interval <1...1000)
        self.set_correct_units()
        # Adjust the number to a reasonable value (1,2,3,5,10,20,30...),
        # modify the lenght-in-pixels accordingly, and change units if needed.
        if   self.number > 750: self.adjust_length_to(1000)
        elif self.number > 350: self.adjust_length_to( 500)
        elif self.number > 250: self.adjust_length_to( 300)
        elif self.number > 150: self.adjust_length_to( 200)
        elif self.number >  50: self.adjust_length_to( 100)
        elif self.number >  35: self.adjust_length_to(  50)
        elif self.number >  25: self.adjust_length_to(  30)
        elif self.number >  15: self.adjust_length_to(  20)
        elif self.number > 7.5: self.adjust_length_to(  10)
        elif self.number > 3.5: self.adjust_length_to(   5)
        elif self.number > 2.5: self.adjust_length_to(   3)
        elif self.number > 1.5: self.adjust_length_to(   2)
        else                  : self.adjust_length_to(   1)
        # Correct units in case we achieved the value of 1000.
        self.set_correct_units()
        # Round number (for nicer output: 1 um instead of 1.0 um)
        self.number = round(self.number)
        # Round pixels (for correct drawing in image = whole pixels)
        self.pixels = round(self.pixels)
