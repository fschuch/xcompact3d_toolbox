from traitlets import BaseDescriptor, Int, Float, Bool, Unicode, List
import re

BOOLEAN_STATES = {".true.": True, ".false.": False}

class TypeMixin(BaseDescriptor):

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class X3dBool(Bool, TypeMixin):
    def from_string(self, s: str) -> None:
        return super().from_string(s.strip("."))

    @classmethod
    def identifier(self, s: str) -> bool:
        return s.lower() in BOOLEAN_STATES


class X3dInt(Int, TypeMixin):

    @classmethod
    def identifier(self, s: str) -> bool:
        return re.fullmatch(r"[-\+]?\d+", s)


class X3dFloat(Float, TypeMixin):

    @classmethod
    def identifier(self, s: str) -> bool:
        if X3dInt.identifier(s):
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False


class X3dUnicode(Unicode, TypeMixin):
    def from_string(self, s: str) -> None:
        return super().from_string(s.strip("'"))

    @classmethod
    def identifier(self, s: str) -> bool:
        return s.startswith("'") and s.endswith("'")


class X3dList(List, TypeMixin):

    @classmethod
    def identifier(self, s: str) -> bool:
        return s.startswith("[") and s.endswith("]")
