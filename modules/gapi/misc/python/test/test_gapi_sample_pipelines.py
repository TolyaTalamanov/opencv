#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
         ('ocl'    , cv.gapi.core.ocl.kernels()),
         ('cpu'    , cv.gapi.core.cpu.kernels()),
         ('fluid'  , cv.gapi.core.fluid.kernels())
         # ('plaidml', cv.gapi.core.plaidml.kernels())
     ]

def custom_add(img1, img2, dtype):
        return cv.add(img1, img2)

def custom_mean(img):
    return cv.mean(img)

def custom_split3(img):
    # NB: cv.split return list but g-api requires tuple in multiple output case
    return tuple(cv.split(img))


class gapi_sample_pipelines(NewOpenCVTests):

    # NB: This test check multiple outputs for operation
    def test_mean_over_r(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # # OpenCV
        _, _, r_ch = cv.split(in_mat)
        expected = cv.mean(r_ch)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        g_out = cv.gapi.mean(r)
        comp = cv.GComputation(g_in, g_out)

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')

    def test_custom_mean(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)
        # sz = (3, 3, 3)
        # in_mat = np.full(sz, 45, dtype=np.uint8)

        # OpenCV
        expected = cv.mean(in_mat)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.mean(g_in)

        comp = cv.GComputation(g_in, g_out)

        pkg    = cv.kernels((custom_mean, 'org.opencv.core.math.mean'))
        actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))

        # Comparison
        self.assertEqual(expected, actual)


    def test_custom_add(self):
        sz = (3, 3)
        in_mat1 = np.full(sz, 45, dtype=np.uint8)
        in_mat2 = np.full(sz, 50 , dtype=np.uint8)

        # OpenCV
        expected = cv.add(in_mat1, in_mat2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        def custom_add(img1, img2, dtype):
                return cv.add(img1, img2)

        pkg = cv.kernels((custom_add, 'org.opencv.core.math.add'))
        actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_multiple_custom_kernels(self):
        sz = (3, 3, 3)
        in_mat1 = np.full(sz, 45, dtype=np.uint8)
        in_mat2 = np.full(sz, 50 , dtype=np.uint8)

        # OpenCV
        expected = cv.mean(cv.split(cv.add(in_mat1, in_mat2))[1])

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_sum = cv.gapi.add(g_in1, g_in2)
        g_b, g_r, g_g = cv.gapi.split3(g_sum)
        g_mean = cv.gapi.mean(g_b)

        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_mean))


        pkg = cv.kernels((custom_add   , 'org.opencv.core.math.add'),
                         (custom_mean  , 'org.opencv.core.math.mean'),
                         (custom_split3, 'org.opencv.core.transform.split3'))

        actual = comp.apply(cv.gin(in_mat1, in_mat2), args=cv.compile_args(pkg))

        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
