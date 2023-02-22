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
        subimages = (image[rowins[0]:rowins[1], colins[0]:colins[2]],
                     image[rowins[0]:rowins[2], colins[2]:colins[3]],
                     image[rowins[2]:rowins[3], colins[1]:colins[3]],
                     image[rowins[1]:rowins[3], colins[0]:colins[1]])
        compressed.append([encode(subimage, quality) for subimage in subimages])
    return image.shape, n_layers, compressed

def encode(image, quality):
    return cv2.imencode('.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

def decode(image):
    return cv2.imdecode(np.frombuffer(image, dtype = np.uint8),
                        cv2.IMREAD_GRAYSCALE)

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

def get_size(jpegs):
    size = sys.getsizeof(jpegs[0])
    for subimages in jpegs[1:]:
        for subimage in subimages:
            size += sys.getsizeof(subimage)
    return size + 4 + 1

def main(args = sys.argv[1:]):
    import time

    # path_image = pathlib.Path(args[0])
    qualitites = (50, 30, 15, 5)

    x = np.linspace(0, 1, 128*(2*len(qualitites) - 1))
    x = np.sin(x*31*3.1416)
    z = np.outer(x, x)
    original_image = (z * 255).astype(np.uint8)
    print(original_image.shape)
    t0 = time.perf_counter()
    shape, layers, jpegs = compress(original_image, qualitites)
    print(f'Time to compress {(time.perf_counter() - t0)*1000:.1f} ms')

    t0 = time.perf_counter()
    compressed_image = decompress(shape, layers, jpegs)
    print(f'Time to decompress {(time.perf_counter() - t0)*1000:.1f} ms')

    sizes = ([sys.getsizeof(jpegs[0])]
             + [sys.getsizeof(subimages[0]) for subimages in jpegs[1:]])
    print(sizes)

    simple_jpeg = encode(original_image, 15)
    simple_compressed = decode(simple_jpeg)

    size_foveated = get_size(jpegs)
    size_simple = sys.getsizeof(simple_jpeg)

    print(size_foveated)
    print(size_simple)
    if '--show' in args:

        import matplotlib.pyplot as plt
        _, axs = plt.subplots(1,3, sharex=True, sharey=True) 
        axs[0].imshow(original_image, cmap = 'gray')
        axs[0].set_title('Original ')
        axs[1].imshow(compressed_image, cmap = 'gray')
        axs[1].set_title(f'Foveated Compression: {size_foveated/1000} kB')
        axs[2].imshow(simple_compressed, cmap = 'gray')
        axs[2].set_title(f'Simple Compression: {size_simple/1000} kB')

        for ax in axs:
            ax.tick_params(labelcolor = 'none', which = 'both',
                            top = False, bottom = False,
                            left = False, right = False)
        plt.show()

if __name__ == '__main__':
    main()