"""Methods to load the CamVid dataset into memory."""
from .camvid import CamVid
from . import videos


# explicitly define the outward facing API of this package
__all__ = [CamVid.__name__, videos.__name__]
