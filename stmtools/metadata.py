"""Metadata schema module.

Now the schema is implemented as annotations without enforcement.
This is to leave future possibilities to enforce typing. (Aug 24, 2023)
"""


from collections.abc import Iterable
from typing import Literal, TypedDict


class STMMetaData(TypedDict, total=False):
    """Type annotations for metadata."""

    techniqueId: str
    datasetId: str
    crs: str | int
    globalAttrib: str
    datasetAttrib: str
    techniqueAttrib: str
    obsDataKeys: Iterable[str]
    auxDataKeys: Iterable[str]
    pntAttribKeys: Iterable[str]
    epochAttribKeys: Iterable[str]


DataVarTypes = Literal[
    "obsData", "auxData", "pntAttrib", "epochAttrib"
]  # Data variable types annotations
