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


class gapi_core_test(NewOpenCVTests):

    def test_add(self):
        # TODO: Extend to use any type and size here
        sz = (1280, 720)
        in1 = np.random.randint(0, 100, sz).astype(np.uint8)
        in2 = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = in1 + in2

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        for pkg in pkgs:
            actual = comp.apply(in1, in2, args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_mean(self):
        sz = (1280, 720, 3)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = cv.mean(in_mat)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.mean(g_in)
        comp = cv.GComputation(g_in, g_out)

        for pkg in pkgs:
            actual = comp.apply(in_mat, args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_split3(self):
        sz = (1280, 720, 3)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        b_ref,g_ref,r_ref = cv.split(in_mat)

        # G-API
        g_in = cv.GMat()
        b,g,r = cv.gapi.split3(g_in)

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(b, g, r))

        for pkg in pkgs:
            b_actual, g_actual, r_actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(b_ref, b_actual, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(g_ref, g_actual, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(r_ref, r_actual, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
