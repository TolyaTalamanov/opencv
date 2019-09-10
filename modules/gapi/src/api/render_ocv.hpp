#include <vector>
#include "render_priv.hpp"

#ifndef OPENCV_RENDER_OCV_HPP
#define OPENCV_RENDER_OCV_HPP

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
void GAPI_EXPORTS mosaic(cv::Mat mat, const cv::Rect &rect, int cellSz);
void GAPI_EXPORTS image(cv::Mat mat, int x, int y, cv::Mat img, cv::Mat alpha);
void GAPI_EXPORTS poly(cv::Mat mat, std::vector<cv::Point>, cv::Scalar color, int lt, int shift);

// FIXME only for tests
void GAPI_EXPORTS drawPrimitivesOCVYUV(cv::Mat &yuv, const Prims &prims);
void GAPI_EXPORTS drawPrimitivesOCVBGR(cv::Mat &bgr, const Prims &prims);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_RENDER_OCV_HPP
