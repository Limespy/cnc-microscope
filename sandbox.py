import pathlib
import sys
import time

import CameraApp as app
import numpy as np
from matplotlib import pyplot as plt

PATH_BASE = pathlib.Path(__file__).parent
path_test = PATH_BASE / 'src' / 'CameraApp' / 'test_images'
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
    vmax = 2**12 - app.default_black_level - 15
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

    images = images[:,:,:,1:3] - app.default_black_level# Greens
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
def DB_log() -> int:
    table = np.array((0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31),
                     dtype= np.uint32)
    import numba as nb
    @nb.njit(nb.types.Tuple((nb.uint32, nb.uint32))(nb.uint32))
    def _DB_log(v):
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v, table[np.uint32(v * 0x07C4ACDD) >> 27]

    n = 5978
    print(_DB_log(n), np.log2(n))
    return 0
# ======================================================================
def exposure_weight() -> int:
    over = 2**12 -1
    values = np.linspace(0, over, dtype = np.uint16, num = 1000)
    b = 2**16-1
    c = 2**4 # Maximum weight
    mn = np.log2(over* over / 4 / b)
    m = np.uint16(np.round(mn/2))
    n = np.uint16(np.ceil(mn/2))
    s = np.uint16(np.ceil(np.log2(b / c)))
    print(m)
    print(n)
    print(s)

    def weight2(v):
        return ((v >> m) * ((over - v) >> n)) >> s

    def weight4(v):
        e = 16 # sqrt((over * over) / b) = over / sqrt(b)
        f = 255 # over / e = sqrt(b)
        g = 128 # sqrt(over * e / 4) = sqrt(over * over / sqrt(b) / 4)
        n_g = 7 # log2(g)
        mid = (f -  (v >> n_g) * ((over - v) >> n_g))
        print(np.min(mid))
        return ((b - mid * mid) >> s)

    def weight4b(v):
        v = v / over
        diff = 1 - v
        result = 16 * v*v*diff * diff
        print(np.max(result))
        return (np.uint16(b * result) >> s)

    def weight3(v):
        v = v / over
        diff = 1 - v
        result = 27/4 * v*v*diff
        print(np.max(result))
        return (np.uint16(b * result) >> s)

    def weight3b(v):
        v = v / over
        diff = 1 - v
        result = 27/4 * v*diff*diff
        print(np.max(result))
        return (np.uint16(b * result) >> s)

    # def weight4(v):
    #     v1 = values / over
    #     p1 = 128 * values # 8 * b / over
    #     p2 = values * (128 - b * 24 * values / over)
    #     v2 = v1*v1
    #     v3 = v2 * v1
    #     v4 = v3 * v1
    #     result = - b * 16 * v4 + b * 32 * v3 - b * 24 * v2 + p1
    #     return np.uint16(result) >> s

    plt.plot(values, weight2(values))
    plt.plot(values, weight3(values))
    plt.plot(values, weight3b(values))
    plt.plot(values, weight4(values))
    plt.plot(values, weight4b(values))
    plt.show()
    return 0
# ======================================================================
def int_sqrt() -> int:

    import numba as nb
    # ------------------------------------------------------------------
    @nb.njit(nb.types.Tuple((nb.uint8, nb.uint16))(nb.uint16))
    def _uint16_sqrt_branchless(s: np.uint16) -> np.uint16:
        '''
        TODO Look into OpenCL clz
        leading = clz(s)
        '''
        s |= 1
        leading = 16 - np.uint8(np.log2(s)) # stand-in for clz
        # print(leading)
        power = ((16 - leading) >> 1)
        # print(power)
        x1 = (1 << power) + (s >> (power + 2))
        # print(x1)
        # x1 = 128 + (s >> 9)
        x1 = (x1 + s // x1) >> 1
        # print(x1)
        x1 = (x1 + s // x1) >> 1
        # print(x1)
        x1 = (x1 + s // x1) >> 1
        # print(x1)
        x1 = (x1 + s // x1) >> 1
        # print(x1)
        x1 = (x1 + s // x1) >> 1
        # print(x1)
        # x1 = (x1 + s // x1) >> 1
        # x1 = (x1 + s // x1) >> 1
        return 6, x1
    # ------------------------------------------------------------------
    @nb.njit(nb.types.Tuple((nb.uint8, nb.uint16))(nb.uint16))
    def _uint16_sqrt(s: np.uint16) -> np.uint16:

        s |= 1
        x0 = 256
        leading = 16 - np.uint8(np.log2(s)) # stand-in for clz
        power = ((16 - leading) >> 1) + 1
        x0 = 1 << power
        # x1 = 1 << (power - 1) + (s >> (power + 1))
        # x0 = 127 + (s >> 2)
        # x0 = (s >> 1) + 1
        x1 = (x0 + s // x0) >> 1
        n: np.uint8 = 1
        while x1 < x0:
            n += 1
            x0, x1 = x1, (x1 + s // x1) >> 1
        return n, x1
    # ------------------------------------------------------------------
    print(_uint16_sqrt_branchless(np.uint16(0)))
    print(_uint16_sqrt_branchless(np.uint16(1)))
    # print(_uint16_sqrt_branchless(np.uint16(2)))
    # print(_uint16_sqrt_branchless(np.uint16(3)))
    # print(_uint16_sqrt_branchless(np.uint16(4)))
    print(_uint16_sqrt_branchless(np.uint16(2**16-1)))
    squares = np.arange(2**16, dtype = np.uint16)
    results = np.array([[n, *_uint16_sqrt_branchless(n)] for n in squares])
    n_arr = results[:, 1]

    print(results[:9])
    errs = np.fabs(results[1:, 1] / np.sqrt(squares)[1:] - 1)

    i_max_err = np.argmax(errs)
    print(results[i_max_err, :], errs[i_max_err])
    i_n_max = np.argmax(n_arr)
    n_max = n_arr[i_n_max]
    n_mean = np.mean(n_arr)
    print(n_mean)
    print(n_max)
    print(n_arr)
    print(results[i_n_max, :])
    print(results[4356])
    # calculating waste
    waste_fraction = n_max/n_mean
    print(f'waste {waste_fraction}')
    return 0
# ======================================================================
def histograms():
    for path_image in (path_test).glob('*.raw'):
        exposure = float(path_image.stem)
        data = app.load_raw(path_image)
        green1 = app.substract_black(app.extract_green1(data), 256)
        hist, edges = np.histogram(green1 / exposure, bins = 256)
        plt.plot((edges[1:] + edges[:-1] / 2), hist)
    plt.xlim(0, (2**12-1) / 0.008)
    plt.show()

# ======================================================================
def main(args = sys.argv[1:]) -> int:
    if not args:
        return 0
    getattr(sys.modules[__name__], args[0])(*args[1:])
    return 0
# ----------------------------------------------------------------------
if __name__ == '__main__':
    raise SystemExit(main())
