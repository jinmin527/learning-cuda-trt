
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
#include <app_yolo/yolo.hpp>
#include <app_http/http_server.hpp>

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

class LogicalController : public Controller{
	SetupController(LogicalController);

public:
	bool startup();
 
public: 
	DefRequestMapping(detect);

private:
    shared_ptr<Yolo::Infer> yolo_;
};

Json::Value LogicalController::detect(const Json::Value& param){

    auto session = get_current_session();
    if(session->request.body.empty())
        return failure("Request body is empty");

    // if base64
    // iLogger::base64_decode();
	cv::Mat imdata(1, session->request.body.size(), CV_8U, (char*)session->request.body.data());
    cv::Mat image = cv::imdecode(imdata, 1);
    if(image.empty())
        return failure("Image decode failed");

    auto boxes = yolo_->commit(image).get();
    Json::Value out(Json::arrayValue);
    for(int i = 0; i < boxes.size(); ++i){
        auto& item = boxes[i];
        Json::Value itemj;
        itemj["left"] = item.left;
        itemj["top"] = item.top;
        itemj["right"] = item.right;
        itemj["bottom"] = item.bottom;
        itemj["class_label"] = item.class_label;
        itemj["confidence"] = item.confidence;
        out.append(itemj);
    }
    return success(out);
}

bool LogicalController::startup(){
    yolo_ = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    return yolo_ != nullptr;
}

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model(){

    if(exists("yolov5s.trtmodel")){
        printf("yolov5s.trtmodel has exists.\n");
        return true;
    }

    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "yolov5s.onnx",
        "yolov5s.trtmodel"
    );
    INFO("Done.");
    return true;
}

int start_http(int port = 9090){

    INFO("Create controller");
	auto logical_controller = make_shared<LogicalController>();
	if(!logical_controller->startup()){
		INFOE("Startup controller failed.");
		return -1;
	}

	string address = iLogger::format("0.0.0.0:%d", port);
	INFO("Create http server to: %s", address.c_str());

	auto server = createHttpServer(address, 32);
	if(!server)
		return -1;
    
    server->verbose();

	INFO("Add controller");
	server->add_controller("/api", logical_controller);

    // 这是一个vue的项目
	// server->add_controller("/", create_redirect_access_controller("./web"));
	// server->add_controller("/static", create_file_access_controller("./"));
	INFO("Access url: http://%s", address.c_str());

	INFO(
		"\n"
		"访问如下地址即可看到效果:\n"
		"1. http://%s/api/detect              使用自定义写出内容作为response\n",
		address.c_str()
	);

	INFO("按下Ctrl + C结束程序");
	return iLogger::while_loop();
}

int main(){

    // 新的实现
    if(!build_model()){
        return -1;
    }
    return start_http();
}