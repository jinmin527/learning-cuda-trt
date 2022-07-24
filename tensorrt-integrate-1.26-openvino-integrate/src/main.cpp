
// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include "kiwi-logger.hpp"
#include <opencv2/opencv.hpp>
#include "kiwi-infer.hpp"
#include "kiwi-app-alphaposev2.hpp"
#include "kiwi-app-scrfd.hpp"
#include "kiwi-app-fall.hpp"
#include "kiwi-app-yolov5.hpp"

using namespace std;

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// hsv转bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

void inference(){

    kiwi::set_backend(kiwi::Backend::OpenVINO);
    auto alphapose = alphaposev2::create_infer("sppe.onnx");
    auto scrfd = scrfd::create_infer("scrfd_2.5g_bnkps.onnx");
    auto fall = fall::create_infer("fall_bp.onnx");
    auto yolo = yolov5::create_infer("yolov5s.onnx");

    auto image = cv::imread("car.jpg");
    auto yoloboxes = yolo->commit(image).get();
    auto faces = scrfd->commit(image).get();

    for(int i = 0; i < yoloboxes.size(); ++i){
        auto& ibox = yoloboxes[i];

        if(ibox.class_label == 0){
            auto box = cv::Rect(ibox.left, ibox.top, ibox.right-ibox.left, ibox.bottom-ibox.top);
            auto keys = alphapose->commit(image, box).get();
            for(auto p:keys){
                if(p.z > 0.1)
                    cv::circle(image, cv::Point(p.x, p.y), 5, cv::Scalar(0, 255, 0), -1, 16);
            }
            
            auto state = fall->commit(keys, box).get();
            INFO("姿态：%s: %f", fall::state_name(get<0>(state)), get<1>(state));
        }

        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(ibox.class_label);
        cv::rectangle(image, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 3);

        auto name      = cocolabels[ibox.class_label];
        auto caption   = cv::format("%s %.2f", name, ibox.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(ibox.left-3, ibox.top-33), cv::Point(ibox.left + text_width, ibox.top), color, -1);
        cv::putText(image, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }

    for(auto& b : faces){
        cv::rectangle(image, cv::Point(b.left, b.top), cv::Point(b.right, b.bottom), cv::Scalar(0, 255, 0), 3);
    }
    cv::imwrite("image-draw.jpg", image);
}

int main(){
    inference();
    return 0;
}