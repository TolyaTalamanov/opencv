#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include "api/render_priv.hpp"

void cv::gapi::wip::draw::render(cv::Mat &bgr,
                                 const cv::gapi::wip::draw::Prims &prims,
                                 const cv::gapi::GKernelPackage& pkg)
{
    cv::GMat in;
    cv::GArray<Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::GRenderBGR::on(in, arr)));
    comp.apply(cv::gin(bgr, prims), cv::gout(bgr), cv::compile_args(pkg));
}

void cv::gapi::wip::draw::render(cv::Mat &y_plane,
                                 cv::Mat &uv_plane,
                                 const Prims &prims,
                                 const GKernelPackage& pkg)
{
    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::GRenderNV12::on(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));
    comp.apply(cv::gin(y_plane, uv_plane, prims),
               cv::gout(y_plane, uv_plane),
               cv::compile_args(pkg));
}

void cv::gapi::wip::draw::splitNV12TwoPlane(const cv::Mat& yuv,
                                                  cv::Mat& y_plane,
                                                  cv::Mat& uv_plane)
{
    y_plane.create(yuv.size(), CV_8UC1);
    uv_plane.create(yuv.size() / 2, CV_8UC2);

    // Fill Y plane
    for (int i = 0; i < yuv.rows; ++i)
    {
        const uchar *in  = yuv.ptr<uchar>(i);
        uchar *out = y_plane.ptr<uchar>(i);
        for (int j = 0; j < yuv.cols; j++)
        {
            out[j] = in[3 * j];
        }
    }

    // Fill UV plane
    for (int i = 0; i < uv_plane.rows; i++)
    {
        const uchar *in = yuv.ptr<uchar>(2 * i);
        uchar *out = uv_plane.ptr<uchar>(i);
        for (int j = 0; j < uv_plane.cols; j++)
        {
            out[j * 2] = in[6 * j + 1];
            out[j * 2 + 1] = in[6 * j + 2];
        }
    }
}

void cv::gapi::wip::draw::BGR2NV12(const cv::Mat &bgr,
                                   cv::Mat &y_plane,
                                   cv::Mat &uv_plane)
{
    GAPI_Assert(bgr.size().width  % 2 == 0);
    GAPI_Assert(bgr.size().height % 2 == 0);

    cv::Mat cpy;
    cvtColor(bgr, cpy, cv::COLOR_BGR2YUV);
    cv::gapi::wip::draw::splitNV12TwoPlane(cpy, y_plane, uv_plane);
}
