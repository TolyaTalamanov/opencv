#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp> // Kernel API's

#include "api/render_ocv.hpp"

namespace cv
{
namespace gapi
{

namespace ocv
{

GAPI_OCV_KERNEL(GOCVRenderNV12, cv::gapi::wip::draw::GRenderNV12)
{
    static void run(const cv::Mat& y, const cv::Mat& uv, const cv::gapi::wip::draw::Prims& prims,
                    cv::Mat& out_y, cv::Mat& out_uv)
    {
        /* FIXME How to render correctly on NV12 format ?
         *
         * Rendering on NV11 via OpenCV looks like this:
         *
         * y --------> 1)NV12toYUV -> yuv -> 2)draw -> yuv -> 3)split -------> out_y
         *                  ^                                     |
         *                  |                                     |
         * uv --------------                                      `----------> out_uv
         *
         *
         * 1) Collect yuv mat from two planes, uv plain in two times less than y plane
         *    so, upsample uv in tow times, with nearest neighbor interpolation
         *
         * 2) Render primitives on YUV
         *
         * 3) Convert yuv to NV12 (Here we can lose color, due uv downsampling)
         *
         */

        auto NV12ToYUV = [](const cv::Mat& y_pln, const cv::Mat& uv_pln, cv::Mat& yuv_pln)
        {
            yuv_pln.create(y_pln.size(), CV_8UC3);

            for (int i = 0; i < uv_pln.rows; ++i)
            {
                const uchar* uv_line = uv_pln.ptr<uchar>(i);
                for (int k = 0; k < 2; ++k)
                {
                    const uchar* y_line   = y_pln.ptr<uchar>(i * 2 + k);
                    uchar* yuv_line = yuv_pln.ptr<uchar>(i * 2 + k);
                    for (int j = 0; j < uv_pln.cols; ++j)
                    {
                        yuv_line[j * 2 * 3    ] = y_line [2 * j    ];
                        yuv_line[j * 2 * 3 + 1] = uv_line[2 * j    ];
                        yuv_line[j * 2 * 3 + 2] = uv_line[2 * j + 1];

                        yuv_line[j * 2 * 3 + 3] = y_line [2 * j + 1];
                        yuv_line[j * 2 * 3 + 4] = uv_line[2 * j    ];
                        yuv_line[j * 2 * 3 + 5] = uv_line[2 * j + 1];
                    }
                }
            }
        };

        cv::Mat yuv;
        NV12ToYUV(y, uv, yuv);

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims);
        cv::gapi::wip::draw::splitNV12TwoPlane(yuv, out_y, out_uv);
    }
};

GAPI_OCV_KERNEL(GOCVRenderBGR, cv::gapi::wip::draw::GRenderBGR)
{
    static void run(const cv::Mat&, const cv::gapi::wip::draw::Prims& prims, cv::Mat& out)
    {
        cv::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims);
    }
};

cv::gapi::GKernelPackage kernels()
{
    static const auto pkg = cv::gapi::kernels<GOCVRenderNV12, GOCVRenderBGR>();
    return pkg;
}

} // namespace ocv

namespace wip
{
namespace draw
{

void mosaic(cv::Mat mat, const cv::Rect &rect, int cellSz)
{
    auto mos = mat(rect);
    cv::Mat tmp;
    cv::resize(mos, tmp, {mos.size().width/cellSz, mos.size().height/cellSz}, 0, 0, cv::INTER_LINEAR);
    cv::resize(tmp, mos, {rect.width, rect.height}, 0, 0, cv::INTER_LINEAR);
};

void image(cv::Mat mat, int x, int y, cv::Mat img, cv::Mat alpha)
{
    auto submat = mat(cv::Rect(x, y, img.size().width, img.size().height));
    cv::Mat alp;
    cv::multiply(alpha, img, alp);
    alp.copyTo(submat);
};

void poly(cv::Mat mat, std::vector<cv::Point> points, cv::Scalar color, int lt, int shift)
{
    std::vector<std::vector<cv::Point>> pp{points};
    cv::fillPoly(mat, pp, color, lt, shift);
};

struct BGR2YUVConverter
{
    cv::Scalar cvtColor(const cv::Scalar& bgr) const
    {
        int y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
        int u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
        int v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;

        cv::Scalar yuv_color(y, u, v);

        return yuv_color;
    }

    void cvtImg(const cv::Mat& in, cv::Mat& out) { cv::cvtColor(in, out, cv::COLOR_BGR2YUV); };
};

struct EmptyConverter
{
    cv::Scalar cvtColor(const cv::Scalar& bgr)   const { return bgr; };
    void cvtImg(const cv::Mat& in, cv::Mat& out) const { out = in;   };
};

// FIXME util::visitor ?
template <typename ColorConverter>
void drawPrimitivesOCV(cv::Mat &in, const Prims &prims)
{
    ColorConverter converter;
    for (const auto &p : prims)
    {
        switch (p.index())
        {
            case Prim::index_of<Rect>():
            {
                const auto& t_p = cv::util::get<Rect>(p);
                const auto color = converter.cvtColor(t_p.color);
                cv::rectangle(in, t_p.rect, color , t_p.thick, t_p.lt, t_p.shift);
                break;
            }

            case Prim::index_of<Text>():
            {
                const auto& t_p = cv::util::get<Text>(p);
                const auto color = converter.cvtColor(t_p.color);
                cv::putText(in, t_p.text, t_p.org, t_p.ff, t_p.fs,
                            color, t_p.thick, t_p.lt, t_p.bottom_left_origin);
                break;
            }

            case Prim::index_of<Circle>():
            {
                const auto& c_p = cv::util::get<Circle>(p);
                const auto color = converter.cvtColor(c_p.color);
                cv::circle(in, c_p.center, c_p.radius, color, c_p.thick, c_p.lt, c_p.shift);
                break;
            }

            case Prim::index_of<Line>():
            {
                const auto& l_p = cv::util::get<Line>(p);
                const auto color = converter.cvtColor(l_p.color);
                cv::line(in, l_p.pt1, l_p.pt2, color, l_p.thick, l_p.lt, l_p.shift);
                break;
            }

            case Prim::index_of<Mosaic>():
            {
                const auto& l_p = cv::util::get<Mosaic>(p);
                mosaic(in, l_p.mos, l_p.cellSz);
                break;
            }

            case Prim::index_of<Image>():
            {
                const auto& i_p = cv::util::get<Image>(p);

                cv::Mat img;
                converter.cvtImg(i_p.img, img);

                image(in, i_p.x, i_p.y, img, i_p.alpha);
                break;
            }

            case Prim::index_of<Poly>():
            {
                const auto& p_p = cv::util::get<Poly>(p);
                const auto color = converter.cvtColor(p_p.color);
                poly(in, p_p.points, color, p_p.lt, p_p.shift);
                break;
            }

            default: cv::util::throw_error(std::logic_error("Unsupported draw operation"));
        }
    }
}

void drawPrimitivesOCVBGR(cv::Mat &in, const Prims &prims)
{
    drawPrimitivesOCV<EmptyConverter>(in, prims);
}

void drawPrimitivesOCVYUV(cv::Mat &in, const Prims &prims)
{
    drawPrimitivesOCV<BGR2YUVConverter>(in, prims);
}

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv
