
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include "box.hpp"

using namespace std;
using namespace cv;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

static std::vector<uint8_t> load_file(const string& file){

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

vector<Box> cpu_decode(float* predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f){
    
    vector<Box> boxes;
    int num_classes = cols - 5;
    for(int i = 0; i < rows; ++i){
        float* pitem = predict + i * cols;
        float objness = pitem[4];
        if(objness < confidence_threshold)
            continue;

        float* pclass = pitem + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < confidence_threshold)
            continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }

    std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b){return a.confidence > b.confidence;});
    std::vector<bool> remove_flags(boxes.size());
    std::vector<Box> box_result;
    box_result.reserve(boxes.size());

    auto iou = [](const Box& a, const Box& b){
        float cross_left   = std::max(a.left, b.left);
        float cross_top    = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) 
                         + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < boxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < boxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = boxes[j];
            if(ibox.label == jbox.label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return box_result;
}

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

vector<Box> gpu_decode(float* predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f){
    
    vector<Box> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    float* output_device = nullptr;
    float* output_host = nullptr;
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold, 
        nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));

    int num_boxes = min((int)output_host[0], max_objects);
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]
            );
        }
    }
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}

int main(){

    auto data = load_file("predict.data");
    auto image = cv::imread("input-image.jpg");
    float* ptr = (float*)data.data();
    int nelem = data.size() / sizeof(float);
    int ncols = 85;
    int nrows = nelem / ncols;
    auto boxes = gpu_decode(ptr, nrows, ncols);
    
    for(auto& box : boxes){
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }
    
    cv::imwrite("image-draw.jpg", image);
    return 0;
}