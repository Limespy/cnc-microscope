import limecamera as lc
from limecamera._aux import PATH_TEST_IMAGES
from limedev.test import BenchmarkResultsType
from limedev.test import eng_round
from limedev.test import run_timed
# ======================================================================
IMAGES_RAW = lc.load_HDR(PATH_TEST_IMAGES/ 'shiny')[0]
GREEN1, GREEN1_OVER = lc.hdr.subtract_black(IMAGES_RAW, 'green1')
# ======================================================================
_hdr_f32 = lc.hdr.hdr_f32
# ----------------------------------------------------------------------
def hdr_loop():
    _hdr_f32(GREEN1, GREEN1_OVER)
# ======================================================================
_hdr_f32_no_tmp = lc.hdr.hdr_f32_simple
# ----------------------------------------------------------------------
def hdr_no_tmp():
    _hdr_f32_no_tmp(GREEN1, GREEN1_OVER)
# ======================================================================
_hdr_f32_simple = lc.hdr.hdr_f32_simple
# ----------------------------------------------------------------------
def hdr_simple():
    _hdr_f32_simple(GREEN1, GREEN1_OVER)
# ======================================================================
def main() -> BenchmarkResultsType:
    results = {}
    for name, function in (('hdr', hdr_loop),
                           ('hdr_simple', hdr_simple),
                           ('hdr_no_tmp', hdr_no_tmp)):
        result, prefix = eng_round(run_timed(function))
        results[f'{name} [{prefix}s]'] = result
    return lc.__version__, results
