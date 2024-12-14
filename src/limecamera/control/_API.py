import pathlib
import subprocess
from typing import TYPE_CHECKING

import numpy as np

from .._aux import RAMDrive
from .._aux import RAW_COLS
from .._aux import RAW_FILE_COLS
from .._aux import RAW_ROWS
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from ..typing_extra import Float64Array
    from ..typing_extra import UInt8Array

else:
    Float64Array = UInt8Array = object
# ======================================================================
PATH_BASE = pathlib.Path(__file__).parent
PATH_CWD = pathlib.Path.cwd()
FILE_EXTENSION = '.rpiraw'
DEFAULT_GAIN = 1.0
DEFAULT_SHUTTER = 1e-3
# ======================================================================
def hello(path: pathlib.Path = PATH_CWD / 'image.jpeg',
          gain: float = DEFAULT_GAIN,
          shutter_s: float = DEFAULT_SHUTTER) -> pathlib.Path:
    subprocess.run(('libcamera-still',
                    '--shutter', str(int(shutter_s * 1e6)),
                    '--gain', str(gain),
                    '-o', str(path)))
    return path
# ======================================================================
def raw(path: pathlib.Path = PATH_CWD / f'image.rpiraw',
        gain: float = DEFAULT_GAIN,
        shutter_s: float = DEFAULT_SHUTTER) -> pathlib.Path:
    subprocess.run(('libcamera-raw', '1' '--rawfull', '--segment', '1', '--flush'
                    '--shutter', str(int(shutter_s * 1e6)),
                    '--gain', str(gain),
                    '-o', str(path)))
    return path
# ======================================================================
def hdr_batch(path_folder: pathlib.Path = PATH_CWD / f'hdr',
              mid_shutter_s: float = DEFAULT_SHUTTER,
              n_images: int = 5):
    low_shutter_s = mid_shutter_s / 2**(n_images // 2)
    _shutter_s = low_shutter_s
    with RAMDrive(size_MiB = n_images * 19) as tmppath:
        for _ in range(n_images):
            raw(tmppath / f'{_shutter_s}.rpiraw', shutter_s = _shutter_s)
            _shutter_s *= 2.
        # Moving from RAM drive to target
        path_folder.mkdir(exist_ok = True)
        for path_image in tmppath.iterdir():
            path_image.replace(path_folder / path_image.name)
# ======================================================================
def load_raw(path: pathlib.Path,
             rows: int = RAW_ROWS,
             cols: int = RAW_COLS,
             stride: int = RAW_FILE_COLS) -> UInt8Array:
    return np.fromfile(path, dtype = np.uint8, count = rows * stride
                       ).reshape((rows, stride))[:,:cols]
# ======================================================================
def load_HDR(path: pathlib.Path,
             rows: int = RAW_ROWS,
             cols: int = RAW_COLS,
             stride: int = RAW_FILE_COLS,
             extension: str = FILE_EXTENSION
             ) -> tuple[UInt8Array, Float64Array]:
    """_summary_

    Parameters
    ----------
    path : pathlib.Path
        _description_
    rows : int, optional
        _description_, by default RAW_ROWS
    cols : int, optional
        _description_, by default FULL_COLS
    stride : int, optional
        _description_, by default RAW_COLS
    extension : str, optional
        _description_, by default FILE_EXTENSION

    Returns
    -------
    tuple[list[UInt8Array], list[float]]
        (raws, exposures)
        raws: images from longest exposure to shortest
        exposures: soreted list of exposures
    """

    meta = [(path, float(path.stem)) for path in path.glob('*' + extension)]
    meta.sort(key = lambda m: m[1], reverse = True)

    raws = np.zeros((len(meta), rows, cols), dtype = np.uint8)

    for index, (path, _) in enumerate(meta):
        raws[index] = load_raw(path, rows, cols, stride)

    return raws, np.fromiter((m[1] for m in meta), dtype = np.float64)
