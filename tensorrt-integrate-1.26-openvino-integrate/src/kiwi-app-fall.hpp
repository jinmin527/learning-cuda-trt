#ifndef KIWI_APP_FALL_HPP
#define KIWI_APP_FALL_HPP

#include "kiwi-common-box.hpp"
#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
#include <memory>
#include <string>

namespace fall{

    typedef std::tuple<std::vector<cv::Point3f>, cv::Rect> Input;

    enum class FallState : int{
        Fall      = 0,
        Stand     = 1,
        UnCertain = 2
    };

    const char* state_name(FallState state);

    class Infer{
    public:
        virtual std::shared_future<std::tuple<FallState, float>> commit(const std::vector<cv::Point3f>& pose16, const cv::Rect& box) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string& engine_file
    );
}; // namespace kiwi

#endif // KIWI_APP_FALL_HPP