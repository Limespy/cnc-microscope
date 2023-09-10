try:
    import API as CameraApp
except ImportError:
    from . import API as CameraApp

import sys

args = sys.argv[1:]

if not args or args[0].lower() == 'hello':
    print('Hello')
    if len(args) >= 2:
        CameraApp.hello(shutter_s = float(args[1]))
    else:
        CameraApp.hello()
elif args[0] == 'image':
    with CameraApp.RAMDrive() as path:
        fpath = CameraApp.take_raw(path)
        print('Loading image')
        image = CameraApp.load_raw(fpath)
        print('Showing image')
        CameraApp.show(image, vmax = 12)
elif args[0] == 'show':
    CameraApp.show(*args[1:])
elif args[0] == 'hdr':
    if len(args) >= 2:
        image = CameraApp.HDR5()
    else:
        image = CameraApp.HDR5()
    image = CameraApp.HDR5()
    CameraApp.show(image, vmax = 2)
