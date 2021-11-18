r"""
Define custom Leaspy exceptions for better downstream handling.

Exceptions classes are nested so to handle in the most convenient way for users::

                    Exception
                        |
                  LeaspyException
                       / \
         TypeError    /   \     ValueError
             |       /     \        |
      LeaspyTypeError      LeaspyInputError
                          /    |    |      \
                         /     |    |  LeaspyIndividualParamsInputError
                        /      |    |
    LeaspyDataInputError       |  LeaspyAlgoInputError
                               |
                    LeaspyModelInputError

For I/O operations, non-Leaspy specific errors may be raised, in particular:
    * :class:`FileNotFoundError`
    * :class:`NotADirectoryError`
"""

class LeaspyException(Exception):
    """Base of all Leaspy exceptions."""
    pass

class LeaspyTypeError(LeaspyException, TypeError):
    """Leaspy Exception, deriving from `TypeError`."""
    pass

class LeaspyInputError(LeaspyException, ValueError):
    """Leaspy Exception, deriving from `ValueError`."""
    pass

class LeaspyDataInputError(LeaspyInputError):
    """Leaspy Input Error for data related issues."""
    pass

class LeaspyModelInputError(LeaspyInputError):
    """Leaspy Input Error for model related issues."""
    pass

class LeaspyAlgoInputError(LeaspyInputError):
    """Leaspy Input Error for algorithm related issues."""
    pass

class LeaspyIndividualParamsInputError(LeaspyInputError):
    """Leaspy Input Error for individual parameters related issues."""
    pass

