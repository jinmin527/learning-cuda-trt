#include "kiwi-face-recognize.hpp"
#include "kiwi-logger.hpp"

namespace kiwi{

    using namespace std;
    using namespace cv;

    void FaceRecognize::clear(){
        all_features_.clear();
        all_ids_.clear();
        feature_length_ = 0;
    }

    bool FaceRecognize::add_item(int idd, const cv::Mat& feature){
        vector<float> ft(feature.cols + 1);
        memcpy(ft.data()+1, feature.ptr<float>(0), feature.cols * sizeof(float));
        ft[0] = feature.cols;
        return add_item(idd, ft.data());
    }

    bool FaceRecognize::add_item(int idd, const float* feature){
        
        int len = check_feature_and_get_length(feature);
        if(len == 0) return false;
        if(feature_length_ == 0) feature_length_ = len;

        all_features_.insert(all_features_.end(), feature+1, feature+1+len);
        all_ids_.insert(all_ids_.end(), idd);
        return true;
    }

    vector<tuple<int, float>> FaceRecognize::query(int topk, float threshold, const cv::Mat& feature){
        vector<float> ft(feature.cols + 1);
        memcpy(ft.data()+1, feature.ptr<float>(0), feature.cols * sizeof(float));
        ft[0] = feature.cols;
        return query(topk, threshold, ft.data());
    }

    vector<tuple<int, float>> FaceRecognize::query(int topk, float threshold, const float* feature){

        int len = check_feature_and_get_length(feature);
        if(len == 0) return {};

        if(topk < 1){
            INFOE("invalid topk %d", topk);
            return {};
        }

        int nitems = all_ids_.size();
        if(topk > nitems) topk = nitems;
        
        cv::Mat feat(nitems, feature_length_, CV_32F, all_features_.data());
        cv::Mat q(feature_length_, 1, CV_32F, (float*)feature + 1);
        auto scores = cv::Mat(feat * q);
        float* ps = scores.ptr<float>(0);
        vector<tuple<int, float>> pairs;
        for(int i = 0; i < nitems; ++i)
            pairs.emplace_back(all_ids_[i], ps[i]);
        
        std::sort(pairs.begin(), pairs.end(), [](tuple<int, float>& a, tuple<int, float>& b){
            return get<1>(a) > get<1>(b);
        });

        if(threshold > 0){
            for(int i = 0; i < pairs.size(); ++i){
                if(get<1>(pairs[i]) < threshold){
                    topk = i;
                    break;
                }
            }
        }
        pairs.resize(topk);
        return pairs;
    }

    int FaceRecognize::check_feature_and_get_length(const float* feature){
        if(feature == nullptr){
            INFOE("feature is nullptr");
            return 0;
        }
        int len = feature[0];
        if(len <= 0 || len > 2048){
            INFOE("invliad feature, len = %d", len);
            return 0;
        }
        if(feature_length_ != 0 && len != feature_length_){
            INFOE("invliad feature, len = %d, mismatch %d", len, feature_length_);
            return 0;
        }
        return len;
    }

};