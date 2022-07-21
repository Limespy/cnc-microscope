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

def RAMdrive(path = default_location, size_MiB: int = 512):
    if path.exists():
        raise FileExistsError('Path already exists')
    path.touch()
    os.system(f'sudo mount -t tmpfs -o size={size_MiB}m images {path}')

def hello():
    os.system('libcamera-hello')

def take_raw(shutter_s: float = 1e-3,
               path: pathlib.Path = default_location,
               fname: str = 'image'):
    os.system(f'libcamera-raw 100 --rawfull --shutter {int(shutter_s * 1e6)} --segment -o {(path / (fname + "%03d." + file_extension))}')
    return path

def parse_bytes(byte1: bytes, byte2: bytes, byte3: bytes
          ) -> tuple[np.uint16, np.uint16]:
    return (((np.uint16(byte1) << 4) | (np.uint16(byte2) >> 4)),
            ((np.uint16(byte2 & 0b00001111) << 8) | np.uint16(byte3)))

def load_raw(path: pathlib.Path = default_location):
    with open(path, 'rb') as raw_file:
        image = []
        for _ in range(height):
            row = []
            for _ in range(width / 3 * 2):
                p1, p2 = parse_bytes(*raw_file.read(3))
                row.append(p1)
                row.append(p2)

            image.append(np.array(row, dtype = np.uint16))
    return np.array(image, dtype = np.uint16)

def show(image):
    plt.ion()
    plt.imshow(image, cmap = 'gray', vmin=0, vmax=2**16-1.)
    plt.show()

