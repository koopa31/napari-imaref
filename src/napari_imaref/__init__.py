__version__ = "1.1.1"

from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._widget import StereotypingWidget, make_stereotyping_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "StereotypingWidget",
    "make_stereotyping_widget",
)
