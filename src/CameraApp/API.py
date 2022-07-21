#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Camera control and image processing application"""
from statistics import mean
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
black_level = 255
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
    os.system(f'libcamera-still --shutter {int(shutter_s * 1e6)} --gain 1 -o {(path / (fname + ".jpeg"))}')

def take_raw(path: pathlib.Path = default_location,
             fname: str = 'image',
             shutter_s: float = exposure):
    fpath = path / (fname + "." + file_extension)
    os.system(f'libcamera-raw 1 --rawfull --segment 1 --flush --shutter {int(shutter_s * 1e6)} --gain 1 -o {fpath}')
    return fpath
# @nb.jit(nopython = True)
def load_raw(path: pathlib.Path = default_location):

    image = np.empty((rows, cols), dtype = np.uint16)
    with open(path, 'rb') as image_file:
        # for rownum in range(rows):
        # row = []
        data = np.fromfile(image_file, dtype = np.uint8, count = rows * stride).reshape(rows, stride)[:,:cols*3 // 2]
        print(data.shape)
        # print(shape)
    bg11 = data[::2, 1::3].astype(np.uint16)
    bg12 = (data[::2, 2::3] & np.uint8(0b00001111)).astype(np.uint16)
    green1 = bg11 << 4 | bg12

    bg21 = data[1::2, ::3].astype(np.uint16)
    bg22 = (data[1::2, 2::3] & np.uint8(0b11110000)).astype(np.uint16)
    green2 = bg21 << 4 | bg22
    image = (green1 + green2) >> 1
    # b1 = data[:, ::3].astype(np.uint16)
    # b2 = data[:, 1::3].astype(np.uint16)
    # b31 = (data[:, 2::3] & np.uint8(0b11110000)).astype(np.uint16)
    # b32 = (data[:, 2::3] & np.uint8(0b00001111)).astype(np.uint16)

    # image[:, ::2] = b1 << 4 | b31
    # image[:, 1::2] = b2 << 4 | b32

    mask = image < black_level
    image[mask] = 0
    image[~mask] -= black_level
    print(f'Mean {np.mean(image)}')
    print(f'Min {np.amin(image)}')
    print(f'Max {np.amax(image)}')
    return image

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

