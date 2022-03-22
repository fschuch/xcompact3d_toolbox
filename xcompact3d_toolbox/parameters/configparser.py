import traitlets
import re
from configparser import RawConfigParser
from .types import BOOLEAN_STATES, TypeMixin


class BaseParser(RawConfigParser):
    # Regular expression for parsing section headers
    _SECT_TMPL = r"&(?P<header>.+)"
    # Compiled regular expression for matching sections
    SECTCRE = re.compile(_SECT_TMPL, re.VERBOSE)
    # Possible boolean values in the configuration
    BOOLEAN_STATES = BOOLEAN_STATES.copy()

    def __init__(self):
        kwargs = dict(
            delimiters="=",
            comment_prefixes=("!", "/End"),
            inline_comment_prefixes=("!"),
            empty_lines_in_values=False,
            allow_no_value=False,
            strict=True,
            interpolation=None,
        )
        super().__init__(**kwargs)

    def write(self, fp):
        fp.write("! -*- mode: f90 -*-\n\n")
        super().write(fp, space_around_delimiters=True)

    def _write_section(self, fp, section_name, section_items, delimiter):
        """Write a single section to the specified `fp'."""
        fp.write("!===================\n")
        fp.write("&{}\n".format(section_name))
        fp.write("!===================\n\n")
        for key, value in section_items:
            value = self._interpolation.before_write(self, section_name, key, value)
            if value is not None or not self._allow_no_value:
                value = delimiter + str(value).replace("\n", "\n\t")
            else:
                value = ""
            fp.write("{}{}\n".format(key, value))
        fp.write("\n")

    def getstring(self, section, option):
        return self.get(section, option).strip("'")

    def to_dict(self):
        return {
            sec: {opt: value for opt, value in self[sec].items()}
            for sec in self.sections()
        }


class ConfigSection(traitlets.HasTraits):

    def __init__(self, name, *args, **kwargs):
        self._name = name
        super(ConfigSection, self).__init__(*args, **kwargs)
    
    @property
    def name(self):
        return self._name


class ConfigParser(traitlets.HasTraits):

    _base_parser = BaseParser
    _base_section = ConfigSection

    _sections = traitlets.Dict(
        key_trait=traitlets.Unicode(),
        value_trait=traitlets.Instance(klass=ConfigSection),
    )

    def sections(self) -> list:
        """Return a list of section names"""
        return list(self._sections.keys())

    def add_section(self, **kwargs):
        pass

    def has_section(self, section: str) -> bool:
        return section in self._sections

    def read_string(self, string: str) -> None:
        config = self._base_parser()
        config.read_string(string)
        self._parse_from_base(config)

    def read_dict(self, dictionary: dict) -> None:
        config = self._base_parser()
        config.read_dict(dictionary)
        self._parse_from_base(config)

    def read_file(self, filename: str) -> None:
        config = self._base_parser()
        with open(filename, "r") as f:
            config.read_file(f)
        self._parse_from_base(config)

    def _parse_from_base(self, config):

        for sec_name, sec_val in config.to_dict().items():
            if sec_name not in self._sections:
                self._sections[sec_name] = self._base_section(name=sec_name)
            for opt_name, opt_val in sec_val.items():
                for cls_type in TypeMixin.get_subclasses():
                    if cls_type.identifier(opt_val):
                        self._sections[sec_name].add_traits(**{opt_name: cls_type(cls_type().from_string(opt_val))})
                        break
                else:
                    raise ValueError(f"Parser can not handle {opt_val}")

    def __getattr__(self, item):
        if item in self._sections:
            return self._sections[item]
        for section in self._sections.values():
            if section.has_trait(item):
                return getattr(section, item)
        raise AttributeError(
            "Class %s does not have a trait named %s" % (self.__class__.__name__, item)
        )
