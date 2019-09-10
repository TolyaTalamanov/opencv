// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_HPP
#define OPENCV_GAPI_RENDER_TESTS_HPP

#include "gapi_tests_common.hpp"
#include "api/render_priv.hpp"
#include "api/render_ocv.hpp"

// FIXME Add more tests
#define rect1 Prim{cv::gapi::wip::draw::Rect{cv::Rect{101, 101, 199, 199}, cv::Scalar{153, 172, 58},  1, LINE_8, 0}}
#define rect2 Prim{cv::gapi::wip::draw::Rect{cv::Rect{100, 100, 199, 199}, cv::Scalar{153, 172, 58},  1, LINE_8, 0}}
#define rect3 Prim{cv::gapi::wip::draw::Rect{cv::Rect{0  , 0  , 199, 199}, cv::Scalar{153, 172, 58},  1, LINE_8, 0}}
#define rect8 Prim{cv::gapi::wip::draw::Rect{cv::Rect{100, 100, 200, 200}, cv::Scalar{153, 172, 58},  1, LINE_8, 0}}
#define box1  Prim{cv::gapi::wip::draw::Rect{cv::Rect{101, 101, 200, 200}, cv::Scalar{153, 172, 58}, -1, LINE_8, 0}}
#define rects Prims{rect1, rect2, rect3, rect8, box1}

// FIXME OSD segmentation fault width > 0 && height > 0
#define rect4 Prim{cv::gapi::wip::draw::Rect{cv::Rect{100, 100, 0, 199  }, cv::Scalar{153, 172, 58},  1,  LINE_8, 0}}
// FIXME OSD segmentation fault x >= 0 && y >= 0
#define rect5 Prim{cv::gapi::wip::draw::Rect{cv::Rect{0  , -1 , 199, 199}, cv::Scalar{153, 172, 58},  1,  LINE_8, 0}}
// FIXME if thick > 1 OpenCV cv::rectangle cuts the corner
#define rect6 Prim{cv::gapi::wip::draw::Rect{cv::Rect{100, 100, 199, 199}, cv::Scalar{153, 172, 58},  10, LINE_8, 0}}
// FIXME OSD segmentation fault
#define box2  Prim{cv::gapi::wip::draw::Rect{cv::Rect{100, 100, 199, 199}, cv::Scalar{153, 172, 58},  -1, LINE_8, 0}}

// FIXME Add more tests (Edge cases)
#define circle1 Prim{cv::gapi::wip::draw::Circle{cv::Point{200, 200}, 100, cv::Scalar{153, 172, 58}, 1, LINE_8, 0}}
#define circle2 Prim{cv::gapi::wip::draw::Circle{cv::Point{10, 30}  , 2  , cv::Scalar{153, 172, 58}, 1, LINE_8, 0}}
#define circle3 Prim{cv::gapi::wip::draw::Circle{cv::Point{75, 100} , 50 , cv::Scalar{153, 172, 58}, 5, LINE_8, 0}}
#define circles Prims{circle1, circle2, circle3}

// FIXME Add more tests (Edge cases)
#define line1 Prim{cv::gapi::wip::draw::Line{cv::Point{50, 50}, cv::Point{250, 200}, cv::Scalar{153, 172, 58}, 1, LINE_8, 0}}
#define line2 Prim{cv::gapi::wip::draw::Line{cv::Point{51, 51}, cv::Point{51, 100}, cv::Scalar{153, 172, 58}, 1, LINE_8, 0}}
#define lines Prims{line1, line2}

// FIXME Add more tests (Edge cases)
// FIXME accuracy failed
#define mosaic1 Prim{cv::gapi::wip::draw::Mosaic{cv::Rect{100, 100, 200, 200}, 5, 0}}
// FIXME accuracy failed
#define mosaics Prims{mosaic1}

#define image1 Prim{cv::gapi::wip::draw::Image{100, 100, getImage(), getAlpha()}}
#define images Prims{image1}

// FIXME Add more tests (Edge cases)
#define polygon1 Prim{cv::gapi::wip::draw::Poly{ {cv::Point{100, 100}, cv::Point{50, 200}, cv::Point{200, 30}, cv::Point{150, 50} }, cv::Scalar{153, 172, 58}, 1, LINE_8, 0} }
#define polygons Prims{polygon1}

// FIXME Add more tests (Edge cases)
#define text1 Prim{cv::gapi::wip::draw::Text{"TheBrownFoxJump", cv::Point{100, 100}, FONT_HERSHEY_SIMPLEX, 2, cv::Scalar{102, 178, 240}, 1, LINE_8, false} }
#define texts Prims{text1}

namespace opencv_test
{

using Prims = cv::gapi::wip::draw::Prims;
using Prim  = cv::gapi::wip::draw::Prim;

namespace
{
    // FIXME avoid this
    cv::Mat getImage()
    {
        return cv::Mat(cv::Size(200, 200), CV_8UC3, cv::Scalar::all(255));
    }

    cv::Mat getAlpha()
    {
        return cv::Mat(cv::Size(200, 200), CV_8UC3, cv::Scalar::all(1));
    }
}

template<class T>
class RenderWithParam : public TestWithParam<T>
{
protected:
    void Init()
    {
        MatType type = CV_8UC3;
        mat_ocv.create(sz, type);
        mat_gapi.create(sz, type);
        cv::randu(mat_ocv, cv::Scalar::all(0), cv::Scalar::all(255));
        mat_ocv.copyTo(mat_gapi);
    }

    cv::Size sz;
    std::vector<cv::gapi::wip::draw::Prim> prims;
    cv::gapi::GKernelPackage pkg;

    cv::Mat y_mat_ocv, uv_mat_ocv, y_mat_gapi, uv_mat_gapi, mat_ocv, mat_gapi;
};

struct RenderNV12 : public RenderWithParam <std::tuple<cv::Size,cv::gapi::wip::draw::Prims,cv::gapi::GKernelPackage>>
{
    void ComputeRef()
    {
        cv::cvtColor(mat_ocv, mat_ocv, cv::COLOR_BGR2YUV);
        cv::gapi::wip::draw::drawPrimitivesOCVYUV(mat_ocv, prims);
        cv::gapi::wip::draw::splitNV12TwoPlane(mat_ocv, y_mat_ocv, uv_mat_ocv);
    }
};

struct RenderBGR : public RenderWithParam <std::tuple<cv::Size,cv::gapi::wip::draw::Prims,cv::gapi::GKernelPackage>>
{
    void ComputeRef()
    {
        cv::gapi::wip::draw::drawPrimitivesOCVBGR(mat_ocv, prims);
    }
};

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_HPP
