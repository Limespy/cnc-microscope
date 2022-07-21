import numpy as np
mask = np.full((2,3,4), False)
array = np.zeros((2,3,4))
marray = np.ma.array(array, mask = mask)
print(marray)
print(marray.mask)

print(marray.mask[:,:,0])