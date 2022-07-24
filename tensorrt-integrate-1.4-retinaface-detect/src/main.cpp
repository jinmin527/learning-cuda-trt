
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

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

using namespace std;

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

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
bool build_model(){

    if(exists("mb_retinaface.trtmodel")){
        printf("mb_retinaface.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("mb_retinaface.onnx", 1)){
        printf("Failed to parse mb_retinaface.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // 配置最小允许batch
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    // 配置最大允许batch
    // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("mb_retinaface.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

vector<unsigned char> load_file(const string& file){
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

struct PriorBox{
    float cx, cy, sx, sy;
};

float desigmoid(float x){
    return -log(1.0f / x - 1.0f);
}

float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

size_t compute_prior_size(int input_width, int input_height, const vector<int>& strides={8, 16, 32}, int num_anchor_per_stage=2){

    int input_area = input_width * input_height;
    size_t total = 0;
    for(int s : strides){
        total += input_area / s / s * num_anchor_per_stage;
    }
    return total;
}

vector<PriorBox> init_prior_box(int input_width, int input_height){

    vector<PriorBox> prior;
    vector<int> strides{8, 16, 32};
    vector<vector<float>> min_sizes{
        vector<float>({16.0f,  32.0f }),
        vector<float>({64.0f,  128.0f}),
        vector<float>({256.0f, 512.0f})
    };
    
    size_t box_count = compute_prior_size(input_width, input_height, strides);
    prior.resize(box_count);

    int prior_row = 0;
    for(int istride = 0; istride < strides.size(); ++istride){
        int stride         = strides[istride];
        auto anchor_sizes  = min_sizes[istride];
        int feature_map_width  = input_width  / stride;
        int feature_map_height = input_height / stride;
        
        for(int y = 0; y < feature_map_height; ++y){
            for(int x = 0; x < feature_map_width; ++x){
                for(int isize = 0; isize < anchor_sizes.size(); ++isize){
                    float anchor_size = anchor_sizes[isize];
                    float dense_cx    = (x + 0.5f) * stride;
                    float dense_cy    = (y + 0.5f) * stride;
                    float s_kx        = anchor_size;
                    float s_ky        = anchor_size;
                    auto& prow       = prior[prior_row++];
                    prow.cx = dense_cx;
                    prow.cy = dense_cy;
                    prow.sx = s_kx;
                    prow.sy = s_ky;
                }
            }
        }
    }
    return prior;
}

struct Face{
    float left, top, right, bottom, confidence;
    float landmark[5][2];
};

void inference(){

    TRTLogger logger;
    auto engine_data = load_file("mb_retinaface.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    if(engine->getNbBindings() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;
    auto prior_boxes = init_prior_box(input_width, input_height);
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    ///////////////////////////////////////////////////
    // letter box
    auto image = cv::imread("group.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float mean[] = {104, 117, 123};
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_b++ = pimage[0] - mean[0];
        *phost_g++ = pimage[1] - mean[1];
        *phost_r++ = pimage[2] - mean[2];
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims = engine->getBindingDimensions(1);
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = 2;
    int output_numel = input_batch * output_numbox * output_numprob;
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // decode box
    vector<Face> bboxes;
    float confidence_threshold = 0.7;

    // 用deconfidence的方式，处理置信度，避免进行softmax的计算，去掉softmax节点
    float deconfidence_threshold = desigmoid(confidence_threshold);
    float nms_threshold = 0.5;
    float variances[] = {0.1f, 0.2f};
    for(int i = 0; i < output_numbox; ++i){
        float* ptr = output_data_host + i * output_numprob;
        float neg_deconfidence = ptr[4];
        float pos_deconfidence = ptr[5];
        float object_deconfidence = (pos_deconfidence - neg_deconfidence);
        if(object_deconfidence < deconfidence_threshold)
            continue;

        auto& prior  = prior_boxes[i];
        float cx     = prior.cx + ptr[0] * variances[0] * prior.sx;
        float cy     = prior.cy + ptr[1] * variances[0] * prior.sy;
        float width  = prior.sx * exp(ptr[2] * variances[1]);
        float height = prior.sy * exp(ptr[3] * variances[1]);
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 对于而分类的置信度而言，可以把softmax转化为sigmoid
        Face face;
        face.confidence = sigmoid(object_deconfidence);
        face.left   = d2i[0] * left   + d2i[2];
        face.right  = d2i[0] * right  + d2i[2];
        face.top    = d2i[0] * top    + d2i[5];
        face.bottom = d2i[0] * bottom + d2i[5];

        float* landmark = ptr + 6;
        for(int j = 0; j < 5; ++j){
            float x = prior.cx + landmark[0] * variances[0] * prior.sx;
            float y = prior.cy + landmark[1] * variances[0] * prior.sy;
            face.landmark[j][0] = d2i[0] * x + d2i[2];
            face.landmark[j][1] = d2i[0] * y + d2i[5];
            landmark += 2;
        }
        bboxes.push_back(face);
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms
    std::sort(bboxes.begin(), bboxes.end(), [](Face& a, Face& b){return a.confidence > b.confidence;});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<Face> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const Face& a, const Face& b){
        float cross_left   = std::max(a.left,   b.left);
        float cross_top    = std::max(a.top,    b.top);
        float cross_right  = std::min(a.right,  b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) 
                         + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& iface = bboxes[i];
        box_result.emplace_back(iface);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(iou(iface, jbox) >= nms_threshold)
                remove_flags[j] = true;
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for(int i = 0; i < box_result.size(); ++i){
        auto& iface = box_result[i];
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(iface.left, iface.top), cv::Point(iface.right, iface.bottom), color, 3);

        for(int j = 0; j < 5; ++j)
            circle(image, cv::Point(iface.landmark[j][0], iface.landmark[j][1]), 3, cv::Scalar(0, 0, 255), -1, 16);

        auto caption   = cv::format("%.2f", iface.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(iface.left-3, iface.top-33), cv::Point(iface.left + text_width, iface.top), color, -1);
        cv::putText(image, caption, cv::Point(iface.left, iface.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("image-draw.jpg", image);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}