from typing import TYPE_CHECKING

import numpy as np

from ..lnumba import nb
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ..typing_extra import Float32Array
    from ..typing_extra import UInt16Array
else:
    Float32Array = UInt16Array = object
# ======================================================================
def highlight(image: Float32Array,
              threshold_low: np.float32,
              threshold_high: np.float32 | None = None
              ) -> Float32Array:
    if threshold_high is None:
        threshold_high = 1. - threshold_low
    image[image <= threshold_low] = threshold_high
    image[image >= threshold_high] = threshold_low
    return image
# ======================================================================
@nb.njit(nb.uint32[:,:](nb.uint16[:,:], nb.uint16[:,:]),
         cache = True, parallel = True)
def interp_checkerboard(arr1_16: UInt16Array, arr2_16: UInt16Array
                        ):
    '''

    Parameters
    ----------
    arr1 : Float32Array
        _description_
    arr2 : Float32Array
        _description_

    Returns
    -------
    Float32Array
        _description_

    Raises
    ------
    ValueError
        _description_
    '''
    # _  1  _  1  _  1  _  1
    # 2  _  2  _  2  _  2  _
    # _  1  _  1  _  1  _  1
    # 2  _  2  _  2  _  2  _
    # _  1  _  1  _  1  _  1
    # 2  _  2  _  2  _  2  _
    # _  1  _  1  _  1  _  1
    # 2  _  2  _  2  _  2  _
    # _  1  _  1  _  1  _  1
    # 2  _  2  _  2  _  2  _
    image = np.empty((arr1_16.shape[0] * 2, arr1_16.shape[1] * 2),
                     dtype = np.uint32)
    arr1 = arr1_16.astype(np.uint32)
    arr2 = arr2_16.astype(np.uint32)
    image[::2, 1::2] = arr1
    image[1::2, ::2] = arr2
    # Corners
    image[0, 0] = (arr1[0, 0] + arr2[0, 0]) // 2
    image[-1, -1] = (arr1[-1, -1] + arr2[-1, -1]) // 2
    # # Edges
    image[0, 2::2] = (arr1[0, :-1] + arr1[0, 1:] + arr2[0, 1:]) // 3
    image[-1, 1:-2:2] = (arr2[-1, :-1] + arr2[0, 1:] + arr1[-1, :-1]) // 3
    image[2::2, 0] = (arr2[:-1, 0] + arr2[1:, 0] + arr1[1:, 0]) // 3
    image[1:-2:2, -1] = (arr1[:-1, -1] + arr1[1:, -1] + arr2[1:, -1]) // 3
    # Middle
    image[1:-1:2, 1:-1:2] = (arr1[:-1,:-1]+ arr1[:-1,1:] + arr2[:-1,:-1] + arr2[:-1,1:]) // 4
    image[2::2, 2::2] = (arr1[1:,:-1] + arr1[1:,1:] + arr2[:-1,1:] + arr2[1:,1:])// 4
    return image
# ======================================================================
def normalise(image, new_max: np.float32) -> Float32Array:
    _min = np.float32(np.amin(image))
    _max = np.float32(np.amax(image))
    new = image.astype(np.float32)
    new -= _min
    new *= np.float32(new_max) / (_max - _min)
    return new
