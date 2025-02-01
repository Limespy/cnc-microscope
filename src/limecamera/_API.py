import pathlib
from time import perf_counter

import numpy as np
from limedev.CLI import get_main
from PIL import Image

from .control import hello
from .control import load_HDR
from .control import load_raw
from .process import extract
from .process import hdr
from .process import subtract_black
from .process.post import normalise
# ======================================================================
def show_raw(path_image: pathlib.Path, channel: str = 'green1') -> int:
    raw = load_raw(path_image)
    image = extract(raw, channel)
    image = normalise(image, np.float32(255.*255.))
    image **= 0.5
    image = image.astype(np.uint8)
    Image.fromarray(image).save(path_image.with_suffix('.png'))
    return 0
# ======================================================================
def hdr_demo(path_folder: pathlib.Path, channel: str = 'green1') -> int:
    """Combines multiple images into single image.

    maximum memory usage should be ~8 arrays * 2 bytes/px * 3 Mpx /
    array =
    """
    raws, exposures = load_HDR(path_folder)

    path_channel = path_folder / channel
    path_channel.mkdir(exist_ok = True)

    for raw, exposure in zip(raws, exposures):
        t0 = perf_counter()
        image = extract(raw, channel)
        print(f'extraction: {perf_counter() - t0:.3f} s')
        print(np.amin(image), np.amax(image))
        image = normalise(image, 255.**2)
        image **= 0.5 # gain
        Image.fromarray(image.astype(np.uint8)
                        ).save(path_channel / f'.{exposure}.png')

    images, high = hdr.subtract_black(raws, channel)
    t0 = perf_counter()
    image = hdr.hdr_f32(images, high)
    print(f'hdr2: {perf_counter() - t0:.3f} s')
    image = normalise(image, 255.**2)
    image **= 0.5 # gain
    Image.fromarray(image.astype(np.uint8)
                    ).save(path_channel / f'.image.png')
    return 0
# ======================================================================
main = get_main(__name__)
