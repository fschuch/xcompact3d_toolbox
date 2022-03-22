import pytest
from hypothesis import given
import hypothesis.strategies as st
from xcompact3d_toolbox.parameters.types import TypeMixin, X3dBool, X3dInt, X3dFloat, X3dUnicode


def test_typemixin_get_subclasses_length():
    assert len(list(TypeMixin.get_subclasses())) > 0

@given(st.booleans())
def test_x3d_bool(value):
    string = f".{value}.".upper()
    assert X3dBool.identifier(string)
    assert X3dBool().from_string(string) == value

@given(st.integers())
def test_x3d_int(value):
    string = str(value)
    assert X3dInt.identifier(string)
    assert X3dInt().from_string(string) == value

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_x3d_float(value):
    string = str(value)
    assert X3dFloat.identifier(string)
    assert X3dFloat().from_string(string) == value

@given(st.text())
def test_x3d_unicode(value):
    string = f"'{value}'"
    assert X3dUnicode.identifier(string)
    assert X3dUnicode().from_string(string) == value