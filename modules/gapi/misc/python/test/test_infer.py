#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class AgeGender(cv.gapi.Network):
    pass



class test_infer(NewOpenCVTests):

    def test_age_gender_infer(self):

        # cv.GMat g_in
        # net = cv.gapi.GNetwork("age-gender", out_shape=(cv.GMAT, cv.GMAT))
        # age, gender = cv.gapi.infer(net)

        # cv.GComputation c(cv.GIn(g_in), cv.GOut(age, gender))

        # cfg = ('path-to-model', 'device')
        # pp = cv.gapi.Params(net, cfg).cfgOutputLayers(['prob', 'age_conv3'])

        # img = cv.imread("path-to-img")
        # age_mat, gender_mat = c.apply(img, cv.compiler_args(pp))

        # cv.GMat g_in
        # cv.GInferInputs inputs
        # inputs['data'] = g_in

        # outputs = cv.gapi.infer('age-gender', inputs)

        # age = outputs['conv3_age']
        # gender = outputs['prob']
        # cv.GComputation c(cv.GIn(g_in), cv.GOut(age, gender))

        # pp = cv.gapi.Params('path-to-model')

        # img = cv.imread('path-to-image')
        # age_mat, gender_mat = c.apply(img, cv.compile_args(cv.gapi.networks(pp)))




if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
