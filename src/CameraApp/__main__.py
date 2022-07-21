try:
    from . import API as CameraApp
except ImportError:
    import API as CameraApp

import sys

args = sys.argv[1:]

if not args:
    CameraApp.hello()
elif args[0] == 'image':
    with CameraApp.RAMdrive() as path:
        CameraApp.take_raw(path)
        image = CameraApp.load_raw(path)
        CameraApp.show(image)