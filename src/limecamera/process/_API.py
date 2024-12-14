import math

import numpy as np
from limedev.CLI import get_main

from . import _aux
from . import hdr
from .._aux import RAW_LIMIT
from . _aux import subtract_black
from .extract import extract
from .post import interp_checkerboard
# ======================================================================
def plot_hdr_weights() -> int:
    from matplotlib import pyplot as plt

    black_level = _aux.DEFAULT_BLACK_LEVELS['green1']
    high = RAW_LIMIT #- black_level
    weight_functions = (hdr.weight3a_u16, hdr.weight4a, hdr.weight4c)

    values = np.arange(high+1, dtype = np.uint16)
    weights = values.copy()
    tmp = values.copy()
    _, (ax_weights, ax_scaled) = plt.subplots(2)
    for weight_function in weight_functions:
        name = weight_function.__name__
        print(name)
        weight_function(values, weights, tmp, high)
        weights += 1 # to handle the special case of all zero
        ax_weights.plot(values, weights, '.', label = name)

        scaled = weights * values
        max_scaled = np.amax(scaled)
        print(f'maximum scaled value {max_scaled}, {math.ceil(math.log2(max_scaled)):.0f}')
        nonzeros = scaled != np.uint16(0)
        ax_scaled.plot(values[nonzeros], np.ceil(np.log2(scaled[nonzeros])),
                       '.', label = name)
    ax_weights.grid()
    ax_weights.legend()
    ax_scaled.grid()
    ax_scaled.legend()
    plt.savefig(_aux.PATH_PACKAGE / '.hdr_weights.pdf')
    return 0
# ======================================================================
main = get_main(__name__)
