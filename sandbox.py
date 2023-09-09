from src import CameraApp as app

from matplotlib import pyplot as plt
import numpy as np
import pathlib
import sys
import time

path_test = pathlib.Path(__file__).parent / 'src' / 'CameraApp' / 'test_images'
vmax = np.uint16(2 ** 12 - 1  - app.default_black_level)
vmin = 0
vrange = vmax - vmin
threshold_fraction = 0.001
threshold = np.uint16(threshold_fraction * vrange)

totalmax = vmax

rows = 3040
cols = 4056
# ======================================================================
def combine(*args):
    _, axs = plt.subplots(6, 1, sharex = True, sharey = True)
    totalmax = vmax
    for i, filepath in enumerate(path_test.glob('*.raw')):
        data = app.load_raw(filepath)#.astype(np.uint32)
        green1 = app.substract_black(app.extract_green1(data), 260)
        # green2 = app.substract_black(app.extract_green2(data), 200)
        image = green1# + green2
        # image = image / np.amax(image)
        image = app.highlight(image, np.amin(image), np.amax(image))
        # mean1 = np.mean(green1)
        # mean2 = np.mean(green2)
        # print(mean1 / mean2)
        # print(np.mean(np.abs(green1 - green2)))
        axs[i].imshow((image / np.amax(image)) ** 0.5, cmap = 'gray')

    t0 = time.perf_counter()
    image = app.process0(path_test)
    # image = image / np.amax(image)
    image = app.highlight(image, np.amin(image), np.amax(image))
    print(f'Time {time.perf_counter() - t0} s')

    # plt.figure(2)

    axs[5].imshow((image / np.amax(image)) ** 0.5, cmap = 'gray')

    plt.show()
# ======================================================================
def color(*args):
    reds = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)
    greens = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)
    blues = np.empty((rows // 2, cols // 2, 5), dtype = np.uint16)

    for n, path_image in zip(range(5), path_test.glob('*.raw')):
        raw = app.load_raw(path_image)
        reds[:,:,n] = app.substract_black(app.extract_red(raw), 260)
        greens[:,:,n] = (app.substract_black(app.extract_green1(raw), 260)
                         + app.substract_black(app.extract_green2(raw), 200)
                         ) // 2
        blues[:,:,n] = app.substract_black(app.extract_blue(raw), 260)
    
    vmax12 = 2**12 -1

    red = app.combine5(reds, np.uint16(vmax12- 260)) * 2.0
    green = app.combine5(greens, np.uint16(vmax12- 200)) * 0.9
    blue = app.combine5(blues, np.uint16(vmax12- 260)) * 1.0

    image = np.array([red, green, blue]).transpose(1, 2, 0)
    image **= 0.5
    plt.imshow(image / np.amax(image))
    plt.figure(2)
    plt.show()
# ======================================================================
def correct_green2(green2):
    vmax = 2**12 - app.black_level - 15
    return np.log(vmax - green2) / np.log(vmax) * 50
# ======================================================================
def interp(*args):
    image1 = np.ones((4,4), dtype = np.float32)
    image2 = image1 * np.float32(2)
    image = app.interp_checkerboard(image1, image2)
    print(np.round(image,3))
# ======================================================================
# analysis
def grey(*args):
    images = np.empty((9, rows // 2, cols // 2, 4), dtype = np.uint16)
    shutters = np.empty((9,1), dtype = np.float32)

    for  n, filepath in enumerate((path_test / 'grey').glob('*.raw')):
        images[n,:,:,:] = app.extract(app.load_raw(filepath))
        shutters[n,:] = np.float32(filepath.stem)

    images = images[:,:,:,1:3] - app.black_level# Greens
    means = np.mean(images, axis = (1, 2))
    stds = np.std(images, axis = (1, 2))
    print(means)
    print(stds/means)
    fig, axs = plt.subplots(3,1)
    axs[0].plot(shutters, means)
    print(shutters)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    shifters = np.arange(8, -1, -1, dtype = np.uint16).reshape(-1,1)
    rescaled = (means.astype(np.uint16)) << shifters
    print(rescaled)
    axs[1].plot(shutters, rescaled)
    axs[1].set_xscale('log')

    offsets = means[:,1] - means[:,0]
    print(offsets)
    axs[2].plot(means[:,1], offsets, label = 'offsets')
    correction = correct_green2(means[:,1])
    corrected = means[:,1] - correction +15
    axs[0].plot(shutters, corrected, label = 'corrected')
    axs[2].plot(means[:,1], correction, label = 'correction')
    axs[2].axhline()
    axs[2].set_xscale('log')
    axs[2].legend()

    if '--show' in args:
        plt.show()
# ======================================================================
def green_diff():
    means = []
    stds = []
    for path_image in (path_test / 'grey').glob('*'):
        image_raw = app.load_raw(path_image)
        image_greens = np.array([app.extract_green1(image_raw),
                                 app.extract_green2(image_raw)]
                                ).astype(np.float32)
        diff = np.diff(image_greens, axis = 0)
        print(diff.shape)
        means.append(np.mean(diff))
        stds.append(np.std(diff))
    print(means)
    print(stds)
# ======================================================================
def compression(path: str):
    image = app.extract_green1(app.load_raw(pathlib.Path(path)))
    normalised = np.divide(image, np.max(image), dtype = np.float16)
    plt.imshow(normalised, cmap = 'gray')
    print(image.shape)
    plt.show()
# ======================================================================
def pixel_combine():
    data = np.array([1420, 2556, 3825, 4000, 4000], dtype = np.uint16)
    print(app.pixel_combine(data, np.uint16(4000)))
# ======================================================================


# ======================================================================
def int_sqrt() -> int:

    import numba as nb
    # ------------------------------------------------------------------
    @nb.njit(nb.types.Tuple((nb.uint8, nb.uint16))(nb.uint16))
    def _int_sqrt(s: np.uint16) -> np.uint16:

        s |= 1
        x0 = 255
        x1 = (x0 + s // x0) >> 1
        n: np.uint8 = 1
        while x1 < x0:
            n += 1
            x0, x1 = x1, (x1 + s // x1) >> 1
        return n, x0
    # ------------------------------------------------------------------
    squares = np.arange(2**16, dtype = np.uint16)
    results = np.array([[n, *_int_sqrt(n)] for n in squares])
    print(results[:9])
    errs = np.fabs(results[1:, 1] / np.sqrt(squares)[1:] - 1)
    i_max_err = np.argmax(errs)
    print(results[i_max_err, :], errs[i_max_err])

    print(_int_sqrt(np.uint16(0)))

    return 0
# ======================================================================
def main(args = sys.argv[1:]) -> int:
    if not args:
        return 0
    getattr(sys.modules[__name__], args[0])(*args[1:])
    return 0
# ----------------------------------------------------------------------
if __name__ == '__main__':
    raise SystemExit(main())