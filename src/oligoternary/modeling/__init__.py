"""Rosetta linker reconstruction and refinement."""

__all__ = ["LinkerModeler"]


def __getattr__(name: str):
    if name == "LinkerModeler":
        from oligoternary.modeling.modeler import LinkerModeler

        return LinkerModeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
