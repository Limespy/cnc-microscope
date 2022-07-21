#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Camera control and image processing application"""
from matplotlib import pyplot as  plt
import numpy as np

import os
import pathlib
path_package = pathlib.Path(__file__).parent.absolute()
file_extension = 'raw'
default_location = path_package / 'tmp_images'

width = 4056
height = 3040

class RAMDrive:

    def __init__(self, path = default_location, size_MiB: int = 512) -> None:
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
        self.path.unlink()

def hello():
    os.system('libcamera-hello')

def take_raw(shutter_s: float = 1e-3,
               path: pathlib.Path = default_location,
               fname: str = 'image'):
    os.system(f'libcamera-raw 100 --rawfull --shutter {int(shutter_s * 1e6)} --segment -o {(path / (fname + "%03d." + file_extension))}')
    return path

def load_raw(path: pathlib.Path = default_location):
    with open(path, 'rb') as raw_file:
        image = []
        for _ in range(height):
            row = []
            for _ in range(width / 3 * 2):
                b1, b2, b3 = raw_file.read(3)
                row.append((np.uint16(b1) << 4) | (np.uint16(b2) >> 4))
                row.append((np.uint16(b2 & 0b00001111) << 8) | np.uint16(b3))

            image.append(np.array(row, dtype = np.uint16))
    return np.array(image, dtype = np.uint16)

def show(image):
    plt.ion()
    plt.imshow(image, cmap = 'gray', vmin=0, vmax=2**16-1.)
    plt.show()

