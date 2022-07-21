from . import API as CameraApp

import sys

args = sys.args[1:]

if not args:
    CameraApp.hello()
elif args[0] == 'image':
    CameraApp.RAMdrive()
    path = CameraApp.take_raw()
    image = CameraApp.load_raw()
    CameraApp.show(image)