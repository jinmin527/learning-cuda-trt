#ifndef KIWI_FACE_RECOGNIZE_HPP
#define KIWI_FACE_RECOGNIZE_HPP

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace kiwi{

    class FaceRecognize{
    public:
        void clear();
        bool add_item(int idd, const cv::Mat& feature);
        bool add_item(int idd, const float* feature);
        std::vector<std::tuple<int, float>> query(int topk, float threshold, const cv::Mat& feature);
        std::vector<std::tuple<int, float>> query(int topk, float threshold, const float* feature);

    private:
        int check_feature_and_get_length(const float* feature);

    private:
        std::vector<float> all_features_;
        std::vector<int> all_ids_;
        int feature_length_ = 0;
    };

}; // kiwi

#endif // KIWI_FACE_RECOGNIZE_HPP