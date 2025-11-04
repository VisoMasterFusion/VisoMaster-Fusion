from typing import Dict, Callable, NewType, TypedDict
from app.helpers.miscellaneous import ParametersDict

LayoutDictTypes = NewType(
    "LayoutDictTypes",
    Dict[str, Dict[str, Dict[str, int | str | list | float | bool | Callable]]],
)

ParametersTypes = NewType("ParametersTypes", ParametersDict)
FacesParametersTypes = NewType("FacesParametersTypes", dict[str, ParametersTypes])

ControlTypes = NewType("ControlTypes", Dict[str, bool | int | float | str])


class MarkerData(TypedDict):
    parameters: FacesParametersTypes
    control: ControlTypes


MarkerTypes = NewType("MarkerTypes", Dict[int, MarkerData])
