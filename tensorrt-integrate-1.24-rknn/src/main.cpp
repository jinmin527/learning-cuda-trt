#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "kiwi-logger.hpp"
#include "kiwi-app-nanodet.hpp"
#include "kiwi-app-scrfd.hpp"
#include "kiwi-app-arcface.hpp"
#include "kiwi-infer-rknn.hpp"
#include "kiwi-yuv.hpp"
#include "kiwi-dpnn.hpp"
#include "kiwi-face-recognize.hpp"

using namespace cv;
using namespace std;

void test_pref(const char* model){

    // 测试性能启用sync mode目的是尽可能的利用npu。实际使用如果无法掌握sync mode，尽量别用
    // 你会发现sync mode的耗时会是非这个模式的80%左右，性能有提升，这与你实际推理时有区别，注意查看
    auto infer = rknn::load_infer(model, true);
    infer->forward();

    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->forward();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("%s avg time: %f ms", model, (toc - tic) / 100);
}

void scrfd_demo(){

    auto infer = scrfd::create_infer("scrfd_2.5g_bnkps.rknn", 0.4, 0.5);
    auto image = cv::imread("faces.jpg");
    auto box_result = infer->commit(image).get();

    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->commit(image).get();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("scrfd time: %f ms", (toc - tic) / 100);

    for(auto& obj : box_result){
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 255, 0), 2);

        auto pl = obj.landmark;
        for(int i = 0; i < 5; ++i, pl += 2){
            cv::circle(image, cv::Point(pl[0], pl[1]), 3, cv::Scalar(0, 0, 255), -1, 16);
        }
    }
    cv::imwrite("scrfd-result.jpg", image);
}

void nanodet_demo(){

    auto infer = nanodet::create_infer("nanodet80-m320x256_nosigmoid.dpnn", 0.4, 0.5, true);
    auto image = cv::imread("dog.jpg");
    auto box_result = infer->commit(image).get();

    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->commit(image).get();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("nanodet time: %f ms", (toc - tic) / 100);

    for(auto& obj : box_result){
        INFO("%f , label = %s", obj.confidence, obj.label_name.c_str());
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("nanodet-result.jpg", image);
}

void arcface_demo(){

    auto det = scrfd::create_infer("scrfd_2.5g_bnkps.rknn", 0.4, 0.5);
    auto ext = arcface::create_infer("w600k_r50_new.rknn");
    auto a = cv::imread("library/2ys3.jpg");
    auto b = cv::imread("library/male.jpg");
    auto c = cv::imread("library/2ys5.jpg");

    auto compute_sim = [](const cv::Mat& a, const cv::Mat& b){
        auto c = cv::Mat(a * b.t());
        return *c.ptr<float>(0);
    };

    auto extract_feature = [&](const cv::Mat& image){
        auto faces = det->commit(image).get();
        if(faces.empty()){
            INFOE("Can not detect any face");
            return cv::Mat();
        }

        auto max_face = std::max_element(faces.begin(), faces.end(), [](kiwi::Face& a, kiwi::Face& b){
            return (a.right - a.left) * (a.bottom - a.top) > (b.right - b.left) * (b.bottom - b.top);
        });

        auto out = arcface::face_alignment(image, max_face->landmark);
        return ext->commit(out, true).get();
    };

    auto fa = extract_feature(a);
    auto fb = extract_feature(b);
    auto fc = extract_feature(c);
    float ab = compute_sim(fa, fb);
    float ac = compute_sim(fa, fc);
    float bc = compute_sim(fb, fc);
    INFO("ab[differ] = %f, ac[same] = %f, bc[differ] = %f", ab, ac, bc);
}

void facerecognize_test(){

    kiwi::FaceRecognize fr;
    
    auto det = scrfd::create_infer("scrfd_2.5g_bnkps.rknn", 0.4, 0.5);
    auto ext = arcface::create_infer("w600k_r50_new.rknn");
    auto a = cv::imread("library/2ys3.jpg");
    auto b = cv::imread("library/male.jpg");
    auto c = cv::imread("library/2ys5.jpg");

    auto compute_sim = [](const cv::Mat& a, const cv::Mat& b){
        auto c = cv::Mat(a * b.t());
        return *c.ptr<float>(0);
    };

    auto extract_feature = [&](const cv::Mat& image){
        auto faces = det->commit(image).get();
        if(faces.empty()){
            INFOE("Can not detect any face");
            return cv::Mat();
        }

        auto max_face = std::max_element(faces.begin(), faces.end(), [](kiwi::Face& a, kiwi::Face& b){
            return (a.right - a.left) * (a.bottom - a.top) > (b.right - b.left) * (b.bottom - b.top);
        });

        auto out = arcface::face_alignment(image, max_face->landmark);
        return ext->commit(out, true).get();
    };

    auto fa = extract_feature(a);
    auto fb = extract_feature(b);
    auto fc = extract_feature(c);

    fr.add_item(0, fa);
    fr.add_item(1, fb);
    fr.add_item(2, fc);
    auto items = fr.query(3, 0.5, fa);
    for(int i = 0; i < items.size(); ++i){
        auto& item = items[i];
        INFO("%d %f", get<0>(item), get<1>(item));
    }
}

void yuv_test(){

    // 高性能的yuv图像缩放处理，
    auto image = cv::imread("dog.jpg");
    cv::resize(image, image, cv::Size(1920, 1080));
    cv::Mat yuv;
    cv::cvtColor(image, yuv, cv::COLOR_BGR2YUV_I420);
    cv::Mat u(image.rows * 0.5, image.cols * 0.5, CV_8U, yuv.ptr<unsigned char>(image.rows));
    cv::Mat v(image.rows * 0.5, image.cols * 0.5, CV_8U, yuv.ptr<unsigned char>(image.rows * 1.25));

    kiwi::YUVImage yv;
    auto tic = kiwi::timestamp_now_float();

    for(int i = 0; i < 100; ++i)
        yv.update(yuv.data, u.data, v.data, image.cols, image.rows);

    auto toc = kiwi::timestamp_now_float();
    INFO("%.2f ms", (toc - tic) / 100);
    cv::imwrite("yuv-test-output.jpg", yv.get_image());
}

int main(){
    // facerecognize_test();
    scrfd_demo();
    nanodet_demo();
    // arcface_demo();
    // yuv_test();
    return 0;
}