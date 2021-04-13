import re
import inspect
from functools import reduce, partial
from typing import Callable, Dict, Iterable, Optional, TypeVar
import warnings

#from typing_extensions import Literal

T = TypeVar('T')
R = TypeVar('R')

def _replace_terms(source: str, mapping: Dict[str, str], flags: int = 0) -> str:
    """
    Replace all occurences of keys in a string by their mapped correspondence.

    <!> The correspondences are searched with word boundaries and is case-sensitive

    Parameters
    ----------
    source
        Source string to replace from
    mapping
        Mapping of terms to replace {original: replacement}
        <!> No replacement term should by an original key to replace
    flags
        Valid flag for :func:`re.sub`

    Example
    -------
    >>> _replace_terms("Can't say the word `less` since I'm wordless. word-", \
                       {'say': 'hear', 'word': '***', 'less': '!?', "I'm": "you're"})
    "Can't hear the *** `!?` since you're wordless. ***-"
    """

    assert len(set(mapping.values()).intersection(mapping.keys())) == 0, "Replacements and replaced should be disjoint."

    return reduce(lambda s, repl: re.sub(fr'\b{re.escape(repl[0])}\b', repl[1], s, flags=flags),
                  mapping.items(), source)

def doc_with_(target: object, original: object, mapping: Dict[str, str] = None, **mapping_kwargs) -> object:
    """
    Document (in-place) a function/class.

    Low-level version of :func:`.doc_with` (refer to its documentation)
    Will set `target.__doc__` in-place (not a decorator function).
    """
    original_doc = original.__doc__
    assert original_doc is not None

    if mapping is not None:
        original_doc = _replace_terms(original_doc, mapping, **mapping_kwargs)

    if hasattr(target, '__func__'): # special method (wrapped) [i.e. classmethod]
        target.__func__.__doc__ = original_doc # in-place modification of wrapped func doc
    else:
        target.__doc__ = original_doc # in-place modification

    return target

def doc_with(original: object, mapping: Dict[str, str] = None, **mapping_kwargs) -> Callable[[object], object]:
    """
    Factory of function/class decorator to use the docstring of `original`
    to document (in-place) the decorated function/class

    Parameters
    ----------
    original: documented Python object
        The object to extract the docstring from
    mapping: dict[str, str], optional
        Optional mapping to replace some terms (case-sensitive and word boundary aware) by others
        from the original docstring.
    **mapping_kwargs:
        Optional keyword arguments passed to :func:`._replace_terms` (flags=...).

    Returns
    -------
    Function/class decorator
    """
    return partial(doc_with_, original=original, mapping=mapping, **mapping_kwargs)

def _get_first_candidate(candidates: Iterable[T], getter: Callable[[T], Optional[R]]) -> Optional[R]:
    for c in candidates:
        obj = getter(c)
        if obj is not None:
            return obj
    return None

def _get_attr_if_cond(attr_name: str, cond: Optional[Callable[[R], bool]] = None) -> Callable[[object], Optional[R]]:
    def getter(obj: object) -> Optional[R]:
        attr: R = getattr(obj, attr_name, None)
        return attr if attr is None or cond is None or cond(attr) else None # lazy bool eval
    return getter

#def doc_with_super(*, if_other_signature: Literal['force', 'warn', 'skip', 'raise'] = 'force', **doc_with_kwargs) -> Callable[[T], T]:
def doc_with_super(*, if_other_signature: str = 'force', **doc_with_kwargs) -> Callable[[T], T]:
    """
    Factory of class decorator that comment (in-place) all of its inherited methods without docstrings + its top docstring
    with the ones from its parent class (the first parent class with this method documented if multiple inheritance)

    Parameters
    ----------
    if_other_signature:
        Behavior if a documented method was found in parent but it has another signature:
            * ``'force'``: patch the method with the found docstring anyway (default)
            * ``'warn'``: patch the method but with a warning regarding signature mismatch
            * ``'skip'``: don't patch the method with the found docstring
            * ``'raise'``: raise a ValueError
    **doc_with_kwargs:
        Optional keyword arguments passed to :func:`.doc_with` (mapping=...).

    Returns
    -------
    Class decorator
    """

    # what methods are we looking to patch with parent doc (including builtin)
    is_method_without_doc = lambda cls_member: inspect.isroutine(cls_member) and cls_member.__doc__ is None

    # simple condition
    member_has_doc = lambda member: member.__doc__ is not None

    # check doc & signature of candidates methods
    def condition_on_super_method_gen(m: Callable) -> Callable[[Callable], bool]:

        # info on subclass method
        m_name = m.__qualname__
        m_sign = inspect.signature(m)

        def condition_on_super_method(super_m: Callable) -> bool:
            # ignore not documented methods
            if super_m.__doc__ is None:
                return False

            sign_is_same = inspect.signature(super_m) == m_sign
            if not sign_is_same:
                if if_other_signature == 'warn':
                    warnings.warn(f'{m_name} has a different signature than its parent {super_m.__qualname__}, patching doc anyway.')
                    return True
                elif if_other_signature == 'raise':
                    raise ValueError(f'{m_name} has a different signature than its parent {super_m.__qualname__}, aborting.')

            # when if_other_signature == 'skip'
            return sign_is_same

        return condition_on_super_method

    # class decorator
    def wrapper(cls):

        assert len(cls.__bases__) > 0, "Must be applied on a class inherinting from others..."

        # patch the class doc itself
        if cls.__doc__ is None:
            super_cls = _get_first_candidate(cls.__bases__, lambda kls: kls if kls.__doc__ is not None else None)
            if super_cls is not None:
                doc_with_(cls, super_cls, **doc_with_kwargs) # in-place

        # get all relevant methods to patch and loop on them: list[(m_name:str, m:method)]
        methods = inspect.getmembers(cls, is_method_without_doc)
        for m_name, m in methods:

            # condition on the super method to be a valid candidate to document method `m`
            if if_other_signature == 'force':
                cond_on_super_method = member_has_doc
            else:
                cond_on_super_method = condition_on_super_method_gen(m)

            super_method = _get_first_candidate(cls.__bases__, _get_attr_if_cond(m_name, cond_on_super_method))

            if super_method is not None:
                doc_with_(m, super_method, **doc_with_kwargs) # in-place

        return cls

    return wrapper
