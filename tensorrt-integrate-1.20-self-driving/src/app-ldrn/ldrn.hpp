#ifndef LDRN_HPP
#define LDRN_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace Ldrn{

    using namespace std;

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<cv::Mat> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<cv::Mat>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid
    );
}; // namespace Yolo

#endif // LDRN_HPP