// This file is part of OpenCV project.  // It is subject to the license terms in the LICENSE file found in the top-level directory // of this distribution and at http://opencv.org/license.html.  //
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

#include <opencv2/gapi/render.hpp>
#include "gapi_render_tests.hpp"

namespace opencv_test
{

TEST_P(RenderNV12, AccuracyTest)
{
    std::tie(sz, prims, pkg) = GetParam();
    Init();

    cv::gapi::wip::draw::BGR2NV12(mat_gapi, y_mat_gapi, uv_mat_gapi);
    cv::gapi::wip::draw::render(y_mat_gapi, uv_mat_gapi, prims, pkg);
    ComputeRef();

    // FIXME move it to comparator
    int not_zero = cv::countNonZero(y_mat_gapi != y_mat_ocv);
    double accuracy = not_zero / static_cast<double>(y_mat_gapi.cols * y_mat_gapi.rows);

    double acc_threshold = 0.008; // difference not more than 0.8 %
    EXPECT_TRUE(std::less<double>()(accuracy, acc_threshold));
}

TEST_P(RenderBGR, AccuracyTest)
{
    std::tie(sz, prims, pkg) = GetParam();
    Init();

    cv::gapi::wip::draw::render(mat_gapi, prims, pkg);
    ComputeRef();

    EXPECT_EQ(0, cv::norm(mat_gapi, mat_ocv));
}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP
