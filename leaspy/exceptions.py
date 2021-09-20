r"""
Define custom Leaspy exceptions for better downstream handling.

Exceptions classes are nested so to handle in the most convenient way for users:


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
LeaspyDataInputError        | LeaspyAlgoInputError
                            |
                LeaspyModelInputError


For I/O operations, non-Leaspy specific errors may be raised, in particular:
    * FileNotFoundError
    * NotADirectoryError
"""

class LeaspyException(Exception):
    pass

class LeaspyTypeError(LeaspyException, TypeError):
    pass

class LeaspyInputError(LeaspyException, ValueError):
    pass

class LeaspyDataInputError(LeaspyInputError):
    pass

class LeaspyModelInputError(LeaspyInputError):
    pass

class LeaspyAlgoInputError(LeaspyInputError):
    pass

class LeaspyIndividualParamsInputError(LeaspyInputError):
    pass

