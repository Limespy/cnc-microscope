import pathlib
import subprocess

import numpy as np
# ======================================================================
PATH_CWD = pathlib.Path.cwd()
PATH_LIBRARY = pathlib.Path(__file__).parent
PATH_TEST_IMAGES = PATH_LIBRARY.parent.parent / 'images' / 'test_images'


RAW_FILE_COLS = 6112
RAW_FILE_ROWS = 3040
FULL_COLS = 4056
FULL_ROWS = 3040
RAW_COLS = 3 * (FULL_COLS // 2)
RAW_ROWS = 3040



RAW_LIMIT = np.uint16((1<<12) - 1)
# ======================================================================
class RAMDrive:
    def __init__(self, path = pathlib.Path('/tmp/ramdisk'), size_MiB: int = 40) -> None:
        if path.exists():
            raise FileExistsError('Path already exists')
        self.path = path
        self.size_MiB = int(size_MiB)
    #───────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.path.mkdir()
        subprocess.run(('sudo', 'mount',
                        '-t', 'tmpfs',
                        '-o', f'size={self.size_MiB}m',
                        'ramdrive', str(self.path)))
        return self.path
    #───────────────────────────────────────────────────────────────────
    def __exit__(self, exc_type, exc_value, traceback):
        subprocess.run(f'sudo umount {self.path}')
        self.path.rmdir()
