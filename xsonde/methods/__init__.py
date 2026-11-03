from xsonde.methods.ceteris_paribus import CeterisParibus
from xsonde.methods.pci import Pci
from xsonde.methods.occlusion import Occlusion
from xsonde.methods.rise import Rise

METHODS = {cls.name: cls for cls in (CeterisParibus, Pci, Occlusion, Rise)}


def register_method(cls):
    METHODS[cls.name] = cls
    return cls


__all__ = ["METHODS", "CeterisParibus", "Occlusion", "Pci", "Rise", "register_method"]
