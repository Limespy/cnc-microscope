import numpy as np
import API as CameraApp
import pathlib
path = pathlib.Path.cwd()
shutter_mean = 2e-3
n = 9
shutter = shutter_mean / 2 ** ((n-1)/2)
for _ in range(n):
    CameraApp.load_raw(path, shutter_s = shutter, fname = str(shutter))
    shutter *= 2
# mask = np.full((2,3,4), False)
# array = np.zeros((2,3,4))
# marray = np.ma.array(array, mask = mask)
# print(marray)
# print(marray.mask)

# print(marray.mask[:,:,0])