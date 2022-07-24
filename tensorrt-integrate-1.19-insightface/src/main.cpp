
#include "app_scrfd/scrfd.hpp"
#include "app_arcface/arcface.hpp"
#include <builder/trt_builder.hpp>
#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>

int main(){

    if(!iLogger::exists("scrfd_2.5g_bnkps.trtmodel")){
        TRT::compile(
            TRT::Mode::FP32,
            1,
            "scrfd_2.5g_bnkps.onnx",
            "scrfd_2.5g_bnkps.trtmodel"
        );
    }

    if(!iLogger::exists("w600k_r50.trtmodel")){
        TRT::compile(
            TRT::Mode::FP32,
            1,
            "../insightface-master/insightface-models/models/buffalo_l/w600k_r50.onnx",
            "w600k_r50.trtmodel"
        );
    }

    auto det = Scrfd::create_infer("scrfd_2.5g_bnkps.trtmodel", 0);
    auto feat = Arcface::create_infer("w600k_r50.trtmodel", 0);

    cv::Mat ruiqiu = cv::imread("ruiqiu.jpg");
    auto ruiqiuface = det->commit(ruiqiu).get()[0];
    auto ruiqiufeat = feat->commit(std::make_tuple(ruiqiu, ruiqiuface.landmark)).get().t();

    cv::Mat image = cv::imread("group.jpg");
    auto faces = det->commit(image).get();
    for(auto& face : faces){
        auto facefeat = feat->commit(std::make_tuple(image, face.landmark)).get();
        float score = cv::Mat(facefeat * ruiqiufeat).at<float>(0);
        INFO("%f", score);

        if(score >= 0.4){
            cv::rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), cv::Scalar(0, 255, 0), 5);

            auto tsize = cv::getTextSize("ruiqiu", 0, 1, 1, 0);
            cv::rectangle(image, cv::Point(face.left-5, face.top-35), cv::Point(face.left+tsize.width+5, face.top-20+tsize.height), cv::Scalar(0, 255, 0), -1);
            cv::putText(image, "ruiqiu", cv::Point(face.left, face.top-10), 0, 1, cv::Scalar(0, 0, 255), 1, 16);
        }else{
            cv::rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), cv::Scalar(0, 255, 255), 2);
        }
        for(int i = 0; i < 5; ++i)
            cv::circle(image, cv::Point(face.landmark[i*2+0], face.landmark[i*2+1]), 3, cv::Scalar(0, 255, 255), -1, 16);
    }
    cv::imwrite("image.jpg", image);
    return 0;
}