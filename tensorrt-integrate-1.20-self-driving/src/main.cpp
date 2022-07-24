
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app-yolo/yolo.hpp>
#include <app-road/road.hpp>
#include <app-ldrn/ldrn.hpp>
#include <app-lane/lane.hpp>

using namespace std;

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

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model(){

    bool success = true;
    if(!exists("yolov5s.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "yolov5s.onnx", "yolov5s.trtmodel");

    if(!exists("road-segmentation-adas.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "road-segmentation-adas.onnx", "road-segmentation-adas.trtmodel");
    
    if(!exists("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx", "ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel");

    if(!exists("new-lane.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "new-lane.onnx", "new-lane.trtmodel");
    return true;
}

static cv::Mat to_render_depth(const cv::Mat& depth){

    cv::Mat mask;
    depth.convertTo(mask, CV_8U, -5, 255);
    //mask = mask(cv::Rect(0, mask.rows * 0.18, mask.cols, mask.rows * (1 - 0.18)));
    cv::applyColorMap(mask, mask, cv::COLORMAP_PLASMA);
    return mask;
}

static void merge_images(
    const cv::Mat& image, const cv::Mat& road,
    const cv::Mat& depth, cv::Mat& scence
){
    image.copyTo(scence(cv::Rect(0, 0, image.cols, image.rows)));

    auto road_crop = road(cv::Rect(0, road.rows * 0.5, road.cols, road.rows * 0.5));
    road_crop.copyTo(scence(cv::Rect(0, image.rows, road_crop.cols, road_crop.rows)));

    auto depth_crop = depth(cv::Rect(0, depth.rows * 0.18, depth.cols, depth.rows * (1 - 0.18)));
    depth_crop.copyTo(scence(cv::Rect(image.cols, image.rows * 0.25, depth_crop.cols, depth_crop.rows)));
}

static void inference(){

    //auto image = cv::imread("imgs/dashcam_00.jpg");
    auto yolov5 = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    auto road = Road::create_infer("road-segmentation-adas.trtmodel", 0);
    auto ldrn = Ldrn::create_infer("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel", 0);
    auto lane = Lane::create_infer("new-lane.trtmodel", 0);

    cv::Mat image, scence;
    cv::VideoCapture cap("4k-tokyo-drive-thru-ikebukuro.mp4");
    float fps = cap.get(cv::CAP_PROP_FPS);
    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    scence = cv::Mat(height * 1.5, width * 2, CV_8UC3, cv::Scalar::all(0));
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'P', 'G', '2'), fps, scence.size());
    //auto scence = cv::Mat(image.rows * 1.5, image.cols * 2, CV_8UC3, cv::Scalar::all(0));

    while(cap.read(image)){
        auto roadmask_fut = road->commit(image);
        auto boxes_fut = yolov5->commit(image);
        auto depth_fut = ldrn->commit(image);
        auto points_fut = lane->commit(image);
        auto roadmask = roadmask_fut.get();
        auto boxes = boxes_fut.get();
        auto depth = depth_fut.get();
        auto points = points_fut.get();
        cv::resize(depth, depth, image.size());
        cv::resize(roadmask, roadmask, image.size());

        for(auto& box : boxes){
            int cx = (box.left + box.right) * 0.5 + 0.5;
            int cy = (box.top + box.bottom) * 0.5 + 0.5;
            float distance = depth.at<float>(cy, cx) / 5;
            if(fabs(cx - (image.cols * 0.5)) <= 200 && cy >= image.rows * 0.85)
                continue;

            cv::Scalar color(0, 255, 0);
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name      = cocolabels[box.class_label];
            auto caption   = cv::format("%s %.2f", name, distance);
            int text_width = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;
            cv::rectangle(image, cv::Point(box.left-3, box.top-20), cv::Point(box.left + text_width, box.top), color, -1);
            cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 0.5, cv::Scalar::all(0), 1, 16);
        }

        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), 
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 0, 0)
        };
        for(int i = 0; i < 18; ++i){
            for(int j = 0; j < 4; ++j){
                auto& p = points[i * 4 + j];
                if(p.x > 0){
                    auto color = colors[j];
                    cv::circle(image, p, 5, color, -1, 16);
                }
            }
        }
        merge_images(image, roadmask, to_render_depth(depth), scence);
        //cv::imwrite("merge.jpg", scence);

        writer.write(scence);
        INFO("Process");
    }
    writer.release();
}

int main(){

    // 新的实现
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}