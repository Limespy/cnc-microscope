import pathlib
from typing import TYPE_CHECKING

import numpy as np
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ..typing_extra import UInt16Array
else:
    UInt16Array = object
# ======================================================================
PATH_PACKAGE = pathlib.Path(__file__).parent
DEFAULT_BLACK_LEVELS = {'red': np.uint16(250),
                        'green1': np.uint16(250),
                        'green2': np.uint16(250),
                        'blue': np.uint16(250)}
# ======================================================================
def subtract_black(image: UInt16Array, black_level: np.uint16) -> UInt16Array:
    """In-place substracts black level."""
    np.maximum(image, black_level, out = image)
    image -= black_level
    return image
# ======================================================================
def average_round(array1: UInt16Array,
                  array2: UInt16Array) -> UInt16Array:
    """In-place averages two arrays with space available with rounding."""
    array1 += array2
    np.bitwise_and(array1, 1, out = array2) # result saved to array2
    array1 >>= 1
    array1 += array2
    return array1
