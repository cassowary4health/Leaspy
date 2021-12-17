from tests import LeaspyTestCase


class ImportTest(LeaspyTestCase):

    def test_import(self):
        def __import():
            import leaspy
            # print(leaspy)
        self.__import_or_fail(__import)

    def test_import_as(self):
        def __import():
            import leaspy as lp
            # print(lp)
        self.__import_or_fail(__import)

    def test_import_main_klass_from(self):
        def __import():
            from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters, __watermark__
            print(__watermark__)
        self.__import_or_fail(__import)

    def __import_or_fail(self, import_function):
        try:
            import_function()
        except Exception as e:
            self.fail(e)
