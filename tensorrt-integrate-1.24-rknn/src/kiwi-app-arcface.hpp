#ifndef KIWI_APP_ARCFACE_HPP
#define KIWI_APP_ARCFACE_HPP

#include "kiwi-common-box.hpp"
#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
#include <memory>
#include <string>

namespace arcface{

    class Infer{
    public:
        virtual std::shared_future<cv::Mat> commit(const cv::Mat& image, bool clone_output=true) = 0;
    };

    // face_landmark xy xy xy xy xy
    cv::Mat face_alignment(const cv::Mat& image, const float face_landmark[10]);

    std::shared_ptr<Infer> create_infer(const std::string& engine_file);
}; // namespace kiwi

#endif // KIWI_APP_ARCFACE_HPP