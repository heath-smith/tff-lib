"""
The exceptions.py module contains custom exception classes used in
the tff_lib.py module. A raised exception propagates up the call stack
to the interpreter then displays a traceback.
"""

class UnitError(Exception):
    """
    The UnitError class inherits from the Exception class and
    should be raised when there is a unit error in a filter
    calculation.

    Attributes
    -----------
    user_val (str): the user input units that caused the exception.\n
    message (str): the explanation of the error.

    Notes
    --------
    ie: if a user enters a unit other than 'rad' or 'deg', a
    UnitError should be raised. If the value is not a string,
    a UnitError should be raised.
    """

    def __init__(self, user_val, message):
        # define instance variables
        self.user_val = user_val
        self.message = message

        # inherit from superclass 'Exception'
        super().__init__(self.message)

    # override the __str__ method
    def __str__(self):
        # return the error message when called
        return f'{self.user_val} --> {self.message}'
