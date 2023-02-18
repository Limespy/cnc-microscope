#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Camera control and image processing application"""
try:
    from .GLOBALS import Float32Array, UInt8Array, UInt16Array
except ImportError:
    from GLOBALS import Float32Array, UInt8Array, UInt16Array
from matplotlib import pyplot as  plt
import numpy as np
import numba as nb
import os
import pathlib
path_package = pathlib.Path(__file__).parent.absolute()
file_extension = 'raw'
default_location = path_package / 'test_images' / 'default.raw'
path_CWD = pathlib.Path.cwd()
stride = 6112
rows = 3040
cols = 4056
exposure = 2e-3
default_black_level = np.uint16(260)
class RAMDrive:

    def __init__(self, path = path_package / '.tmp_images', size_MiB: int = 40) -> None:
        if path.exists():
            raise FileExistsError('Path already exists')
        self.path = path
        self.size_MiB = int(size_MiB)
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.path.mkdir()
        os.system(f'sudo mount -t tmpfs -o size={self.size_MiB}m images {self.path}')
        return self.path
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        os.system(f'sudo umount {self.path}')
        self.path.rmdir()

def hello(path: pathlib.Path = path_CWD,
          fname: str = 'image',
          shutter_s: float = exposure):
    print(f'Shutter time {shutter_s} s')
    os.system(f'libcamera-still --shutter {int(shutter_s * 1e6)} --gain 1 -o {(path / (fname + ".jpeg"))}')

def take_raw(path: pathlib.Path = default_location,
             fname: str = 'image',
             shutter_s: float = exposure):
    fpath = path / (fname + "." + file_extension)
    os.system(f'libcamera-raw 1 --rawfull --segment 1 --flush --shutter {int(shutter_s * 1e6)} --gain 1 -o {fpath}')
    return fpath
#%%═════════════════════════════════════════════════════════════════════
# DEBAYERING
left_half  = np.uint8(0b11110000)
right_half = np.uint8(0b00001111)

@nb.njit(nb.uint16[:,:](nb.uint8[:,:], nb.uint8, nb.uint8, nb.uint8, nb.uint8),
         cache = True)
def _extract(raw_image, row1 , row2, col1, half):
    '''
    Strtucture is:
    [b_h], [g1_h], [g1_l:b_l], ...
    [g2_h], [r_h], [r_l:g2_l], ...
    :                         .
    :                           .
    '''
    B1 = raw_image[row1::2, col1::3]
    B2 = raw_image[row2::2, 2::3] & half
    return (B1 << 4 | B2).astype(np.uint16)

@nb.njit(nb.uint16[:,:](nb.uint8[:,:]), cache = True)
def extract_blue(raw_image: UInt8Array) -> UInt16Array:
    return _extract(raw_image, 0, 0, 0, left_half)

@nb.njit(nb.uint16[:,:](nb.uint8[:,:]), cache = True)
def extract_green1(raw_image: UInt8Array) -> UInt16Array:
    return _extract(raw_image, 0, 0, 1, right_half)

@nb.njit(nb.uint16[:,:](nb.uint8[:,:]), cache = True)
def extract_green2(raw_image: UInt8Array) -> UInt16Array:
    return _extract(raw_image, 1, 0, 1, left_half)

@nb.njit(nb.uint16[:,:](nb.uint8[:,:]), cache = True)
def extract_red(raw_image: UInt8Array) -> UInt16Array:
    return _extract(raw_image, 1, 1, 1, right_half)

def extract(data, channel = None):

    if channel is None:
        image = np.empty((rows // 2, cols // 2, 4), dtype = np.uint16)
        image[:,:,0] = extract_blue(data)
        image[:,:,1] = extract_green1(data)
        image[:,:,2] = extract_green2(data)
        image[:,:,3] = extract_red(data)
        return image
    elif channel == 'blue':
        return extract_blue(data)
    elif channel == 'green1':
        return extract_green1(data)
    elif channel == 'green2':
        return extract_green2(data)
    elif channel == 'red':
        return extract_red(data)

def substract_black(image: UInt16Array, black_level: np.uint16) -> UInt16Array:
    mask = image < black_level
    image[mask] = np.uint16(0)
    image[~mask] -= black_level
    return image

def highlight(image: Float32Array,
              threshold_low: np.float32,
              threshold_high: np.float32 | None= None
              ) -> Float32Array:
    if threshold_high is None:
        threshold_high = 1 - threshold_low
    image[image <= threshold_low] = threshold_high
    image[image >= threshold_high] = threshold_low
    return image


# def debayer(data: UInt8Array, mode: str = 'binning') -> UInt16Array:

#     out = np.empty((()))

def load_raw(path: pathlib.Path = default_location,
             rows = rows,
             cols = cols,
             stride = stride) -> UInt8Array:

    with open(path, 'rb') as image_file:
        data = np.fromfile(image_file, dtype = np.uint8, count = rows * stride).reshape(rows, stride)[:,:(cols*3) >> 1]
    return data

extractors = {'r': extract_red,
              'g1': extract_green1,
              'g2': extract_green2,
              'b': extract_blue}
def show(imagepath: str, channel = 'g1'):
    plt.imshow(extractors[channel](load_raw(pathlib.Path(imagepath))),
               vmax = 2 ** 12 - 1, cmap = 'gray', norm = 'log')
    plt.show()

def HDR5(shutter = 4e-3):
    shutter /= 4
    with RAMDrive() as folder:
        for _ in range(5):
            shutter *= 2
            take_raw(folder, shutter_s = shutter)
        image = process1(folder)
    return image

@nb.jit(nb.uint32[:,:](nb.uint16[:,:], nb.uint16[:,:]),
        nopython = True, cache = True, parallel = True)
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


@nb.njit(nb.uint16(nb.uint16[:], nb.uint16),
         cache = True)
def pixel_combine(data, vmax = 4095):
    '''Combines data datapoints into single pixel
    Data is arranged from smallest to largest with each value being '''
    total = np.uint32(0)
    n_valid = np.uint32(1) # There is always at least one pixel valid
    # Finding first valid pixel
    for i in range(5):
        pixel = data[i]
        if pixel > 0:
            if pixel >= vmax: # In case the first value is overexposed
                return 0xFFFF
            total += (pixel << (4 - i))
            break
    # Looping until overexposed
    for i in range(i+1, 5):
        pixel = data[i]
        if pixel >= vmax: # pixel is overexposed
            break # and all the following ones would be too
        total += (pixel << (4 - i))
        n_valid += 1
    return np.uint16(total // n_valid)

@nb.njit(nb.uint16[:,:](nb.uint16[:, :,:], nb.uint16),
         cache = True)
def combine5(images: UInt16Array,
             vmax: np.uint16) -> UInt16Array:
    '''images[row, column, stack]'''
    n_row, n_col, _ = images.shape
    out = np.empty((n_row, n_col), dtype = np.uint16)
    for irow in range(n_row):
        for icol in range(n_col):
            out[irow, icol] = pixel_combine(images[irow, icol], vmax)
    return out

def correct_green2(green2):
    vmax = 2**16 - default_black_level - 15
    return np.log(vmax - green2) / np.log(vmax) * 50

def process0(path_folder):
    vmax12 = 2**12 -1 
    black_level_g1 = np.uint16(260)
    images1 = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)
    for n, path_image in zip(range(5), path_folder.glob('*.raw')):
        data = load_raw(path_image)
        images1[:,:,n] = substract_black(extract_green1(data), black_level_g1)
    return combine5(images1, vmax = np.uint16((vmax12 - black_level_g1)))

def process1(path_folder: pathlib.Path) -> Float32Array:
    '''Combines multiple images into single image'''
    vmax12 = 2**12 -1
    black_level_g1 = np.uint16(260)
    black_level_g2 = black_level_g1 + 60
    images1 = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)
    images2 = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)
    for n, path_image in zip(range(5), path_folder.glob('*.raw')):
        data = load_raw(path_image)
        images1[:,:, n] = substract_black(extract_green1(data), black_level_g1)
        images2[:,:, n] = substract_black(extract_green2(data), black_level_g2)
        print(f'{np.mean(images1[n,:,:]):.1f}, {np.mean(images2[n,:,:]):.1f}')
    image1 = combine5(images1, vmax = np.uint16((vmax12 - black_level_g1)))
    image2 = combine5(images2, vmax = np.uint16((vmax12 - black_level_g2)))
    return interp_checkerboard(image1, image2)