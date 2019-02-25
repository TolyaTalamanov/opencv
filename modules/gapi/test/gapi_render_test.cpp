// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"
#include "opencv2/gapi/render.hpp"

namespace opencv_test
{

TEST(Render, DrawRectangle)
{
    cv::Mat out_mat(100, 100, CV_8UC3);
    cv::randu(out_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat ref_mat;
    out_mat.copyTo(ref_mat);
    cv::Point2f p1(50, 50), p2(75, 75);
    cv::Scalar color(255, 0, 0);
    auto r = cv::Render::create(cv::Render::BackendType::OCV);

    cv::rectangle(ref_mat, p1, p2, color);

    r->rectangle(p1, p2, color);
    r->run(out_mat);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST(Render, DrawText)
{
    cv::Mat out_mat(100, 100, CV_8UC3);
    cv::randu(out_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat ref_mat;
    out_mat.copyTo(ref_mat);
    cv::Point2f p1(50, 50);
    cv::Scalar color(255, 0, 0);
    std::string text = "Lorem ipsum";

    auto r = cv::Render::create(cv::Render::BackendType::OCV);

    cv::putText(ref_mat, text, p1, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

    r->putText(text, p1, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    r->run(out_mat);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

}
