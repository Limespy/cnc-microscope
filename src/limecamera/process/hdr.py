from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from .._aux import FULL_COLS
from .._aux import FULL_ROWS
from .._aux import RAW_LIMIT
from ..lnumba import IS_CACHE
from ..lnumba import nb
from ..lnumba import nbA
from ..lnumba import nbARO
from ._aux import DEFAULT_BLACK_LEVELS
from ._aux import subtract_black as _subtract_black
from .extract import extract
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ..typing_extra import Float32Array
    from ..typing_extra import UInt16Array
else:
    Float32Array = UInt16Array = object
# ======================================================================
_1_FLOAT32 = np.float32(1.)
_2_FLOAT32 = np.float32(2.)
_1_FLOAT16 = np.float16(1.)
_2_FLOAT16 = np.float16(2.)
_1_UINT16 = np.uint16(1)
_1_UINT16 = np.uint16(2)
# ======================================================================
def full_histogramm(image: UInt16Array):
    bincount = np.bincount(image.flatten())
    values = np.arange(len(bincount), dtype = np.uint16)
    return values, bincount
# ----------------------------------------------------------------------
def compact_histogramm(image: UInt16Array):
    values, bincount = full_histogramm(image)
    nonzero = bincount > 0
    return values[nonzero], bincount[nonzero]
# ======================================================================
def subtract_black(data: UInt16Array, channel: str,
               rows: int = FULL_ROWS,
               cols: int = FULL_COLS) -> tuple[UInt16Array, np.uint16]:
    """_summary_

    Parameters
    ----------
    path_folder : pathlib.Path
        _description_

    Returns
    -------
    tuple[list[UInt16Array], np.uint16]
        (images, high)
        images: images from longest exposure to shortest
        high: where overexposed remapped after black level correction
    """
    black_level = DEFAULT_BLACK_LEVELS[channel]
    images = np.zeros((len(data), rows // 2, cols // 2), dtype = np.uint16)
    for index, raw in enumerate(data):
        extract(raw, channel, rows, cols, out = images[index])
    _subtract_black(images, black_level)
    return images, RAW_LIMIT - black_level
# ======================================================================
def weight0(v, out, *_):
    out[:] = 1.
# ----------------------------------------------------------------------
def weight2(v):
        '''
        2nd degree polynomial of with unit range representation is
        x*(1-x)
        substractions: 1
        bitshifts: 3
        multiplications: 1
        divisions: 0'''
        return ((v >> 3) * ((RAW_LIMIT - v) >> 3)) >> 12
# ----------------------------------------------------------------------
_weight3a_scaler = np.float32(36. * 3**3 / 2**2)
# ----------------------------------------------------------------------
def weight3a_u16(v: UInt16Array, out: UInt16Array, tmp2: UInt16Array,
            high: np.uint16) -> None:
    """

    unit range function is 4 / 27 * x * (1 - x)**2


    Parameters
    ----------
    v : UInt16Array
        _description_
    out : UInt16Array
        _description_
    tmp2 : UInt16Array
        _description_
    high : np.uint16
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _v = v.astype(np.float32)
    _v /= np.float32(high)
    _1_v = _1_FLOAT32 - _v
    _v *= _weight3a_scaler
    _v *= _1_v
    _v *= _1_v
    np.copyto(out, _v, casting = 'unsafe')
# ----------------------------------------------------------------------
# @nb.njit(nb.types.void(nbARO(2, nb.float32),
#                        nbA(2, nb.float32),
#                        nbA(2, nb.float32)),
#          cache = False, parallel = True)
def weight3a_f32(v: Float32Array, out: Float32Array) -> None:
    """

    unit range function is x * (1 - x)**2


    Parameters
    ----------
    v : UInt16Array
        _description_
    out : UInt16Array
        _description_
    tmp2 : UInt16Array
        _description_
    high : np.uint16
        _description_

    Returns
    -------
    _type_
        _description_
    """
    np.subtract(_1_FLOAT32, v, out)
    np.square(out, out)
    out *= v
# ----------------------------------------------------------------------
_weight3a_f16_scaler = np.float16(3**3 / 2**2)
# ----------------------------------------------------------------------
def weight3a_f16(v: Float32Array, out: Float32Array, tmp: Float32Array, high: np.float32) -> None:
    """

    unit range function is 4 / 27 * x * (1 - x)**2


    Parameters
    ----------
    v : UInt16Array
        _description_
    out : UInt16Array
        _description_
    tmp2 : UInt16Array
        _description_
    high : np.uint16
        _description_

    Returns
    -------
    _type_
        _description_
    """
    np.divide(v, high, out)
    np.subtract(_1_FLOAT16, out, tmp)
    out *= _weight3a_f16_scaler
    out *= tmp
    out *= tmp
# ----------------------------------------------------------------------
_weight4a_scaler = np.float32(47. * (4**4 / 3**3))
# ----------------------------------------------------------------------
def weight4a(v: UInt16Array, out: UInt16Array, tmp2: UInt16Array,
            high: np.uint16) -> None:
    """

    unit range function is 4 / 27 * x * (1 - x)**2


    Parameters
    ----------
    v : UInt16Array
        _description_
    out : UInt16Array
        _description_
    tmp2 : UInt16Array
        _description_
    high : np.uint16
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _v = v.astype(np.float32)
    _v /= np.float32(high)
    _1_v = _1_FLOAT32 - _v
    _v *= _weight4a_scaler
    _v *= _1_v
    _v *= _1_v
    _v *= _1_v
    np.copyto(out, _v, casting = 'unsafe')
# ----------------------------------------------------------------------
def weight4c(v: UInt16Array, out: UInt16Array, tmp2: UInt16Array,
             high: np.uint16) -> None:
    '''
    4th degree polynomial of with unit range representation is
    256 / 27 x*(1-x)^3
    substractions: 1
    bitshifts: 3
    multiplications: 3
    divisions: 2'''

    # Complicated due to avoiding memory copy
    print(np.amin(v), np.amax(v))
    np.subtract(high, v, out = out)
    print(np.amin(out), np.amax(out))
    out >>= 4 # round(np.log2(b / over))
    print(np.amin(out), np.amax(out))
    np.right_shift(v, 4, out = tmp2) # round(np.log2(b / over))
    # in-place operations to reduce copying
    tmp2 *= out
    np.square(out, out = out)
    print(np.amin(out), np.amax(out))
    tmp2 //= 83 # round(((over / k1)**4 / (256/27 * b))**(1/2))
    out //= 83 # round(((over / k1)**4 / (256/27 * b))**(1/2))
    print(np.amin(out), np.amax(out))
    out *= tmp2
    print(np.amin(out), np.amax(out))
    out >>= 12 # returning to range [0, 15]
    print(np.amin(out), np.amax(out))
# ----------------------------------------------------------------------
def _divround(array, divisor, tmp):
    """Im-place division with rounding."""
    # Division with rounding to nearest
    np.divmod(array, divisor, out = (array, tmp))
    tmp <<= 1
    # Should be 1 if over half, 0 if under
    tmp //= divisor
    array += tmp
# ======================================================================
def hdr_u16(images: list[UInt16Array], overexposed: np.uint16) -> UInt16Array:

    overexposed = np.float32(overexposed)
    # Setting up storage  array for the sum of weights
    weights_accumulator = np.zeros(images[0].shape, dtype = np.uint16)

    # To save memory, using temporary value arrays
    weights = np.zeros(images[0].shape, dtype = np.uint16)
    tmp = np.zeros(images[0].shape, dtype = np.uint16)

    # fig, (ax_images_all, ax_weights) = plt.subplots(2)


    for i, image in enumerate(images):
        # ax_images_all.plot(*compact_histogramm(image), '.', label = str(i))
        weight3a_u16(image, weights, tmp, overexposed) # stores to weights
        # ax_weights.plot(*compact_histogramm(weights), '.', label = str(i))

        # print(image[375,1500])

        if i == 4:
            # To handle cases where the pixel is under or overexposed
            # in all images, i.e. weight total would be 0,
            # Lowest exposure weights are increased by one.
            # In case of all underexposed, the end results would be 0
            # In case of all overexposed, the end result would be high.
            weights += _1_UINT16

        image *= weights
        # print(f'weighted {i}', np.amax(image))
        weights_accumulator += weights

    weights_accumulator
    # ax_images_all.legend()
    # ax_weights.legend()
    # ax_images_all
    # plt.show()
    # plt.clf()

    # print('accumulator', np.amax(weights_accumulator))

    # Using weights as output
    output = weights
    output[:] = images[0]
    _divround(output, weights_accumulator, tmp)
    # plt.plot(*compact_histogramm(output), '.', label = '0')

    # print('0', np.amax(output))

    for i, image in enumerate(images[1:], start = 1):

        _divround(image, weights_accumulator, tmp)
        # Final shift, i.e. multiplication by factor of the exposure
        image <<= i
        # print(i, np.amax(image))
        # plt.plot(*compact_histogramm(image), '.', label = str(i))
        output += image

    # plt.plot(*compact_histogramm(output), '.', label = 'output')
    # print('out',np.amax(output))
    # plt.legend()
    # plt.show()
    return output
# ======================================================================
# @nb.njit(nb.float32[:,:](nbARO(3, nb.uint16), nb.uint16),
#          cache = IS_CACHE, parallel = True)
def hdr_f32(images: UInt16Array, overexposed: np.uint16) ->Float32Array:
    """Similar to hdr2, but using float32."""

    _overexposed = _1_FLOAT32 / np.float32(overexposed)

    weighted_all = images * _overexposed

    out = weighted_all[0]

    # First iteration is a part of the setup
    weights = _1_FLOAT32 - out
    np.square(weights, weights)
    weights *= out
    weights_accumulator = weights.copy()

    out *= weights

    multiplier = _2_FLOAT32

    i_last = len(images) - 1

    for i, image in enumerate(weighted_all[1:]):
        np.subtract(_1_FLOAT32, image, weights)
        np.square(weights, weights)
        weights *= image
        if i == i_last:
            # To handle cases where the pixel is under or overexposed
            # in all images, i.e. weight total would be 0,
            # Lowest exposure weights are increased by one.
            # In case of all underexposed, the end results would be 0
            # In case of all overexposed, the end result would be high.
            weights += 1e-9
        weights_accumulator += weights
        image *= weights

        image *= multiplier
        out += image

        multiplier *= _2_FLOAT32

    out /= weights_accumulator
    return np.ascontiguousarray(out)
# ======================================================================
def hdr_f32_no_tmp(images: UInt16Array, overexposed: np.uint16) ->Float32Array:
    """Similar to hdr2, but using float32."""

    _overexposed = _1_FLOAT32 / np.float32(overexposed)

    images_all = images * _overexposed # Rescaling

    weights = _1_FLOAT32 - images_all
    weights *= weights
    weights *= images_all

    images_all[-1] += 1e-9

    images_all *= weights

    multiplier = _2_FLOAT32
    for image in images_all[1:]:
        image *= multiplier
        multiplier *= _2_FLOAT32

    out = images_all.sum(0)
    out /= weights.sum(0)
    return out
# ======================================================================
def hdr_f32_simple(images: UInt16Array, overexposed: np.uint16) ->Float32Array:
    _overexposed = _1_FLOAT32 / np.float32(overexposed)

    images_all = images.astype(np.float32)

    _images_all = images_all * _overexposed
    _1_images_all = _1_FLOAT32 - _images_all

    weights = _images_all * _1_images_all * _1_images_all
    weights[-1] += 1e-9

    images_all *= weights

    multiplier = _2_FLOAT32
    for image in images_all[1:]:
        image *= multiplier
        multiplier *= _2_FLOAT32

    return images_all.sum(0) / weights.sum(0)
# ======================================================================
def hdr_f16(images: list[UInt16Array], overexposed: np.uint16) -> UInt16Array:
    """Similar to hdr2, but using float32."""
    # Setting up storage  array for the sum of weights

    weights_accumulator = np.zeros(images[0].shape, dtype = np.float16)
    # To save memory, using temporary value arrays
    weights = np.zeros(images[0].shape, dtype = np.float16)
    tmp = np.zeros(images[0].shape, dtype = np.float16)
    weighted = []
    for i, image in enumerate(images):
        image_f32 = image.astype(np.float16)
        weight3a_f16(image_f32, weights, tmp, overexposed) # stores to weights
        if i == 4:
            # To handle cases where the pixel is under or overexposed
            # in all images, i.e. weight total would be 0,
            # Lowest exposure weights are increased by one.
            # In case of all underexposed, the end results would be 0
            # In case of all overexposed, the end result would be high.
            weights += _1_FLOAT16

        image_f32 *= weights
        weighted.append(image_f32)
        weights_accumulator += weights

    del images

    # Using weights as output
    output = weights
    np.copyto(output, weighted[0])
    output /= weights_accumulator

    multiplier = _2_FLOAT16
    for image in weighted[1:]:

        image /= weights_accumulator
        # Final shift, i.e. multiplication by factor of the exposure
        image *= multiplier
        output += image
        multiplier *= _2_FLOAT16

    return output
# ======================================================================
def hdr_pair_f32(image_low: UInt16Array, image_high: UInt16Array,
                 overexposed: np.uint16,
                 multiplier: np.float32) -> UInt16Array:
    """Similar to hdr2, but using float32."""

    _overexposed = _1_FLOAT32 / np.float32(overexposed)

    # To save memory, using temporary value arrays
    weights_low = np.zeros(image_low.shape, dtype = np.float32)
    weights_high = np.zeros(image_low.shape, dtype = np.float32)
    tmp = np.zeros(image_low.shape, dtype = np.float32)

    image_low_f32 = image_low * _overexposed
    weight3a_f32(image_low_f32, weights_low)
    weights_low += 1e-6
    image_low_f32 *= weights_low
    image_low_f32 *= multiplier


    image_high_f32 = image_high * _overexposed
    weight3a_f32(image_high_f32, weights_high)
    image_high_f32 *= weights_high

    out = image_low_f32
    weights = weights_low

    weights += weights_high

    out += image_high_f32

    out /= weights
    return out
