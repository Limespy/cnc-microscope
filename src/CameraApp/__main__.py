try:
    from . import API as CameraApp
except ImportError:
    import API as CameraApp

import sys

args = sys.argv[1:]

if not args or args[0].lower() == 'hello':
    print('Hello')
    CameraApp.hello()
elif args[0] == 'image':
    with CameraApp.RAMDrive() as path:
        fpath = CameraApp.take_raw(path)
        print('Loading image')
        image = CameraApp.load_raw(fpath)
        print('Showing image')
        CameraApp.show(image, vmax = 12)
elif args[0] == 'show':
    CameraApp.show()
elif args[0] == 'hdr':
    image = CameraApp.HDR5()
    CameraApp.show(image, vmax = 16)