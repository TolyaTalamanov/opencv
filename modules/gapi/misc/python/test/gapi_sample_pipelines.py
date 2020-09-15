#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
         cv.gapi.core.ocl.kernels(),
         cv.gapi.core.cpu.kernels(),
         cv.gapi.core.fluid.kernels()
         # cv.gapi.core.plaidml.kernels()
       ]


class gapi_sample_pipelines(NewOpenCVTests):

    # NB: This test check multiple outputs for operation
    def test_swap_rb(self):
        sz = (100, 100, 3)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)

        # # OpenCV
        b,g,r = cv.split(in_mat)
        expected = cv.merge((r,g,b))

        # G-API
        g_in = cv.GMat()
        b,g,r = cv.gapi.split3(g_in)
        g_out = cv.gapi.merge3(r,g,b)
        comp = cv.GComputation(g_in, g_out)

        actual = comp.apply(in_mat)

        for pkg in pkgs:
            actual = comp.apply(in_mat, args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
