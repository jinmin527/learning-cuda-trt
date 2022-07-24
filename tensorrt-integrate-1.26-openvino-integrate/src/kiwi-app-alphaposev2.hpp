#ifndef KIWI_APP_ALPHAPOSEV2_HPP
#define KIWI_APP_ALPHAPOSEV2_HPP

#include "kiwi-common-box.hpp"
#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
#include <memory>
#include <string>

namespace alphaposev2{

    class Infer{
    public:
        virtual std::shared_future<std::vector<cv::Point3f>> commit(const cv::Mat& image, cv::Rect box) = 0;
    };

    std::shared_ptr<Infer> create_infer(const std::string& engine_file);
}; // namespace kiwi

#endif // KIWI_APP_ALPHAPOSEV2_HPP