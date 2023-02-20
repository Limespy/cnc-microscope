import PIL
import numpy as np
from io import BytesIO
import sys
import cv2


def indices(half, step, n):
    return (half - (2*n + 1) * step,
            half - (2*n - 1) * step,
            half + (2*n - 1) * step,
            half + (2*n + 1) * step)
        
def compress(image, qualities):
    # indices = gen_indices(*image.shape, len(qualities))
    n_layers = len(qualities)
    rows, cols = image.shape
    rowhalf = rows // 2
    colhalf = cols // 2
    rowstep = rowhalf // (n_layers*2 - 1)
    colstep = colhalf // (n_layers*2 - 1)
    # Compress center
    center = encode(image[rowhalf - rowstep:rowhalf + rowstep,
                          colhalf - colstep:colhalf + colstep],
                    qualities[0])
    compressed = [center]
    for i, quality in enumerate(qualities[1:], start = 1):
        rowins = indices(rowhalf, rowstep, i)
        colins = indices(colhalf, colstep, i)
        subimages = [image[rowins[0]:rowins[1], colins[0]:colins[2]],
                     image[rowins[0]:rowins[2], colins[2]:colins[3]],
                     image[rowins[2]:rowins[3], colins[1]:colins[3]],
                     image[rowins[1]:rowins[3], colins[0]:colins[1]]]
        subset = []
        print(rowins)
        for subimage in subimages:
            print(subimage.shape)
            subset.append(encode(subimage, quality))
        compressed.append(subset)
    return image.shape, n_layers, compressed

def encode(image, quality):
    return cv2.imencode('.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]

def decode(image):
    return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

def decompress(shape, n_layers, images):

    rows, cols = shape
    rowhalf = rows // 2
    colhalf = cols // 2
    rowstep = rowhalf // (n_layers*2 - 1)
    colstep = colhalf // (n_layers*2 - 1)
    image = np.empty(shape, dtype = np.uint8)
    # Decompress center
    image[rowhalf - rowstep:rowhalf + rowstep,
          colhalf - colstep:colhalf + colstep] = decode(images[0])
    for n, imageset in enumerate(images[1:], 1):
        rowins = indices(rowhalf, rowstep, n)
        colins = indices(colhalf, colstep, n)
        image[rowins[0]:rowins[1], colins[0]:colins[2]] = decode(imageset[0])
        image[rowins[0]:rowins[2], colins[2]:colins[3]] = decode(imageset[1])
        image[rowins[2]:rowins[3], colins[1]:colins[3]] = decode(imageset[2])
        image[rowins[1]:rowins[3], colins[0]:colins[1]] = decode(imageset[3])
    return image

def main(args = sys.argv[1:]):
    import matplotlib.pyplot as plt
    import pathlib

    # path_image = pathlib.Path(args[0])
    qualitites = (80, 60, 40, 30, 20 ,10)
    x = np.linspace(-1, 1, 56*(2*len(qualitites) - 1))
    z = np.outer(1 - x**2, 1- x**2)
    print(np.amin(z))
    original_image = (z * 255).astype(np.uint8)
    compressed_image = decompress(*compress(original_image, qualitites))

    _, axs = plt.subplots(1,2) 
    axs[0].imshow(original_image, cmap = 'gray')

    axs[1].imshow(compressed_image, cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    main()