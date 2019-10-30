import unittest


class ImportTest(unittest.TestCase):

    def test_import(self):
        def __import():
            import leaspy
            print(leaspy)
        self.__import_or_fail(__import)

    def test_import_as(self):
        def __import():
            import leaspy as lp
            print(lp)
        self.__import_or_fail(__import)

    def __import_or_fail(self, import_function):
        try:
            import_function()
        except Exception as e:
            self.fail(e)
