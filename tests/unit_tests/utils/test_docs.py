import re
import doctest

import leaspy.utils.docs as lsp_docs
from leaspy.utils.docs import doc_with, doc_with_super

class A:

    @classmethod
    def cm(cls):
        """Class method of A"""
        ...

    @staticmethod
    def sm():
        """Static method of A"""
        ...

    def m(self):
        """Method for a A instance"""
        ...

    @classmethod
    def cm_bis(cls):
        """Class method of A (bis)"""
        ...

    @staticmethod
    def sm_bis():
        """Static method of A (bis)"""
        ...

    def m_bis(self):
        """Method for a A instance (bis)"""
        ...

    def foo(self, x):
        """
        Can't say the word `less` since I'm wordless. Word-
        """
        return x

class B:
    """Class B"""

    def foo(self, y):
        """Foo doc from B"""
        ...

    def bar(self):
        """Bar from B"""
        ...

@doc_with_super(mapping={'A':'C', 'B':'C'})
class C(A, B):

    @classmethod
    def cm(cls):
        ...

    @staticmethod
    def sm():
        ...

    def m(self):
        ...

    def foo(self, x):
        return

    def bar(self):
        return

@doc_with(A.foo, {'say': 'hear', 'word': '***', 'less': '!?', "I'm": "you're"}, flags=re.IGNORECASE)
def bar():
    return

def test_doc_with():

    # doc_with
    assert bar.__doc__.strip() == "Can't hear the *** `!?` since you're wordless. ***-"

    # doc_with_super
    assert C.__doc__.strip() == "Class C"

    assert C.foo.__doc__.strip() == "Can't say the word `less` since I'm wordless. Word-"
    assert C.bar.__doc__.strip() == """Bar from C"""

    # not replaced methods (because not re-implemented at all)
    assert C.cm.__doc__.strip() == "Class method of C"
    assert C.sm.__doc__.strip() == "Static method of C"
    assert C.m.__doc__.strip() == """Method for a C instance"""

    # not replaced methods (because not re-implemented at all)
    assert C.cm_bis.__doc__.strip() == "Class method of A (bis)"
    assert C.sm_bis.__doc__.strip() == "Static method of A (bis)"
    assert C.m_bis.__doc__.strip() == """Method for a A instance (bis)"""

    # test all examples contained in docstrings of any object of module
    doctest.testmod(lsp_docs)
