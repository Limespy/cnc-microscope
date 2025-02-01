import limecamera as lc
from limecamera._aux import PATH_TEST_IMAGES
# ======================================================================
IMAGES_RAW = lc.load_HDR(PATH_TEST_IMAGES/ 'shiny')[0]
GREEN1, GREEN1_OVER = lc.hdr.subtract_black(IMAGES_RAW, 'green1')
# ======================================================================
def hdr():
    function = lc.hdr.hdr_f32
    for _ in range(100):
        function(GREEN1, GREEN1_OVER)
