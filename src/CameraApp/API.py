#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Camera control and image processing application"""
from .GLOBALS import Float32Array, UInt8Array, UInt16Array

from matplotlib import pyplot as  plt
import numpy as np
import numba as nb
import os
import pathlib
path_package = pathlib.Path(__file__).parent.absolute()
file_extension = 'raw'
default_location = path_package / 'tmp_images'
path_CWD = pathlib.Path.cwd()
stride = 6112
rows = 3040
cols = 4056
exposure = 2e-3
black_level = np.uint16(260)
class RAMDrive:

    def __init__(self, path = default_location, size_MiB: int = 40) -> None:
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
left_half = np.uint8(0b11110000)
right_half = np.uint8(0b00001111)
@nb.jit(nopython = True, cache = True)
def extract_blue(raw_image: UInt8Array) -> UInt16Array:
    b1 = raw_image[::2, ::3].astype(np.uint16)
    b2 = (raw_image[0::2, 2::3] & left_half).astype(np.uint16)
    return b1 << 4 | b2

@nb.jit(nopython = True, cache= True)
def extract_green1(raw_image: UInt8Array) -> UInt16Array:
    b1 = raw_image[::2, 1::3].astype(np.uint16)
    b2 = (raw_image[::2, 2::3] & right_half).astype(np.uint16)
    return b1 << 4 | b2

@nb.jit(nopython = True, cache = True)
def extract_green2(raw_image: UInt8Array) -> UInt16Array:
    b1 = raw_image[1::2, ::3].astype(np.uint16)
    b2 = (raw_image[1::2, 2::3] & left_half).astype(np.uint16)
    return b1 << 4 | b2

@nb.jit(nopython = True, cache = True)
def extract_red(raw_image: UInt8Array) -> UInt16Array:
    b1 = raw_image[1::2, 1::3].astype(np.uint16)
    b2 = (raw_image[1::2, 2::3] & right_half).astype(np.uint16)
    return b1 << 4 | b2

# @nb.jit(nopython = True)
def substract_black(image: UInt16Array) -> UInt16Array:
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
    image[image < threshold_low] = 1.
    image[image > threshold_high] = 0.
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

def show(image = None, vmax = 16):
    vmax = 2 ** vmax -1
    if image is None:
        image = np.array([[vmax, vmax//2], [vmax//3, vmax // 4]], dtype = np.uint16)
    plt.imshow(image, cmap = 'gray', vmin=0, vmax=vmax)
    plt.show()

def HDR5():
    vmax = 2**12 - 1
    mean_shutter = exposure

    array = np.zeros((5, rows//2, cols//2), dtype = np.uint16)
    mask = np.full(array.shape, False)
    images = np.ma.array(array, mask = mask, dtype = np.uint16)
    del array
    del mask
    with RAMDrive() as folder:
        shutter = mean_shutter * 4
        for shift in range(5):
            shutter /= 2
            path = take_raw(folder, shutter_s = shutter)
            tmp_image = load_raw(path)
            images[shift, :, :] = tmp_image << shift
            images.mask[shift, :, :] = (0 == tmp_image) | (tmp_image == vmax)
            path.unlink()
    image = images.astype(np.uint64).sum(axis = 0) 
    del images
    image_max = np.amax(image)
    image *= np.uint64(0xffff)
    print(f'Max before normalisation {image_max }')
    image //= image_max
    print(f'Vmax {2 ** 16 -1}')
    print(f'Mean {np.mean(image)}')
    print(f'Min {np.amin(image)}')
    print(f'Max {np.amax(image)}')
    print(image.shape)
    return image

def interpolate_greens(green1: Float32Array, green2: Float32Array
                       ) -> Float32Array:
    '''
    
    _  1  _  1  _  1  _  1
    2  _  2  _  2  _  2  _
    _  1  _  1  _  1  _  1
    2  _  2  _  2  _  2  _
    _  1  _  1  _  1  _  1
    2  _  2  _  2  _  2  _
    _  1  _  1  _  1  _  1
    2  _  2  _  2  _  2  _
    _  1  _  1  _  1  _  1
    2  _  2  _  2  _  2  _

    Parameters
    ----------
    green1 : Float32Array
        _description_
    green2 : Float32Array
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
    
    if green1.shape != green2.shape:
        raise ValueError(
            f'Arrays different shapes. {green1.shape} != {green2.shape}')
    image = np.empty((green1.shape[0] * 2, green1.shape[1] * 2),
                     dtype = np.float32)
    # Corners
    image[0, 0] = (green1[0,0] + green2[0,0]) / 2
    
    # Edges
    ## Top
    image[0, 1:-2:2] = 


def combine5(images: UInt16Array,
             threshold_fraction: float = 0.01,
             vmin: np.uint16 | None = None,
             vmax: np.uint16 | None = None,
             ) -> Float32Array:
    if vmin is None: vmin = np.amin(images)
    if vmax is None: vmax = np.amax(images)
    vrange = vmax - vmin
    threshold = np.uint16(threshold_fraction * vrange)
    threshold_low = vmin + threshold
    threshold_high = vmax - threshold
    images_ma = np.ma.array(images,
                            mask = np.full(images.shape, False),
                            dtype = np.uint16)
    if len(images.shape) == 3: # Only one channel
        for i in range(5):
            # inverse, because masked array
            images_ma.mask[i,:,:] |= images_ma[i,:,:] > threshold_high
            images_ma.mask[i,:,:] |= images_ma[i,:,:] < threshold_low
            images_ma[i,:,:] *= np.uint16(6 - i)
    elif len(images.shape) == 4 and images.shape[3] == 3: # three channels
        for i in range(5):
            # inverse, because masked array
            images_ma.mask[i,:,:,:] |= images_ma[i,:,:,:] > threshold_high
            images_ma.mask[i,:,:,:] |= images_ma[i,:,:,:] < threshold_low
            images_ma[i,:,:,:] *= np.uint16(6 - i)
    return np.array(images_ma.mean(axis = 0), dtype = np.float32)