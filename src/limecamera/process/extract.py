from typing import TYPE_CHECKING

import numpy as np

from ..lnumba import nb
from ..lnumba import nbDecC
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ..typing_extra import UInt8Array
    from ..typing_extra import UInt16Array
else:
    UInt8Array = UInt16Array = object
# ======================================================================
right_half = np.uint8(0b00001111)

@nbDecC
def _extract_right(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    '''
    Strtucture is:
    [b_h], [g1_h], [g1_l:b_l], ...
    [g2_h], [r_h], [r_l:g2_l], ...
    :                         .
    :                           .
    '''
    out[:] = raw_image[:, 1::3].astype(np.uint16)
    out <<= np.uint16(4)
    out |= (raw_image[:, 2::3] & right_half).astype(np.uint16)
    return out
# ----------------------------------------------------------------------
@nbDecC
def _extract_left(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    '''
    Strtucture is:
    [b_h], [g1_h], [g1_l:b_l], ...
    [g2_h], [r_h], [r_l:g2_l], ...
    :                         .
    :                           .
    '''
    out[:] = raw_image[:, 0::3].astype(np.uint16)
    out <<= np.uint16(4)
    out |= (raw_image[:, 2::3] >> np.uint8(4)).astype(np.uint16)
    return out
# ----------------------------------------------------------------------
@nbDecC
def extract_blue(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    return _extract_left(raw_image[0::2], out)
# ----------------------------------------------------------------------
@nbDecC
def extract_green1(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    return _extract_right(raw_image[0::2], out)
# ----------------------------------------------------------------------
@nbDecC
def extract_green2(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    return _extract_left(raw_image[1::2], out)
# ----------------------------------------------------------------------
@nbDecC
def extract_red(raw_image: UInt8Array, out: UInt16Array) -> UInt16Array:
    return _extract_right(raw_image[1::2], out)
# ----------------------------------------------------------------------
extractors = {'r': extract_red,
              'g1': extract_green1,
              'g2': extract_green2,
              'b': extract_blue}
# ----------------------------------------------------------------------
def extract(data: UInt8Array,
            channel: str = '',
            rows: int = 3040,
            cols: int = 4056,
            *,
            out: UInt16Array | None = None) -> UInt16Array:
    if not channel:
        if out is None:
            out = np.empty((4, rows // 2, cols // 2), dtype = np.uint16)
        extract_blue(data, out[0,:,:])
        extract_green1(data, out[1,:,:])
        extract_green2(data, out[2,:,:])
        extract_red(data, out[3,:,:])
        return out
    if out is None:
        out = np.empty((rows // 2, cols // 2), dtype = np.uint16)
    if channel == 'blue':
        return extract_blue(data, out)
    if channel == 'green1':
        return extract_green1(data, out)
    if channel == 'green2':
        return extract_green2(data, out)
    if channel == 'red':
        return extract_red(data, out)
