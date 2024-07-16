import pathlib

import matplotlib.pyplot as plt
import numpy as np
# ======================================================================
PATH_BASE = pathlib.Path(__file__).parent
PATH_FIGURES = PATH_BASE / 'figures'
# ======================================================================
def _rectangle_in_annulus(thickness: float, r_outer: float, r_inner: float):

    r_middle2 = r_inner + thickness
    r_middle2 *= r_middle2

    width = 2 * (r_outer * r_outer - r_middle2) ** 0.5

    return width, width * thickness


def main():
    r_tool_mm = 25.
    r_lens_mm = 9.
    l_free_mm = r_tool_mm - r_lens_mm
    A_circle_mm2 = np.pi * (l_free_mm * l_free_mm) * 0.25
    thickness_mm = np.linspace(0., l_free_mm, 100)

    width_mm, area_mm2 = _rectangle_in_annulus(thickness_mm,
                                               r_tool_mm,
                                               r_lens_mm)

    ax = plt.subplot(1, 1, 1)
    ax.set_title(f'Outer radius {r_tool_mm:.1f} mm, '
                 f'Inner radius {r_lens_mm:.1f} mm')
    ax.axhline(A_circle_mm2, color = 'green')
    ax.plot(thickness_mm, area_mm2, color = 'blue', label = 'Area')
    ax.set_xlabel('Thickness [mm]')
    ax.set_ylabel('Area [mm$^2$]')
    ax.grid()
    plt.legend(loc = 'center left')
    ax2 = ax.twinx()
    ax2.plot(thickness_mm, width_mm, color = 'orange', label = 'Width')
    ax2.set_ylabel('Width [mm]')
    plt.legend(loc = 'upper right')
    plt.savefig(PATH_FIGURES / 'spindle_unit_annular.svg',
                bbox_inches = 'tight')
# ======================================================================
if __name__ == '__main__':
    raise SystemExit(main())
