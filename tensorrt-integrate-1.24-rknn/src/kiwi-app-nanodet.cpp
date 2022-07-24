#include "kiwi-app-nanodet.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-infer-rknn.hpp"
#include "kiwi-logger.hpp"
#include "kiwi-dpnn.hpp"

namespace nanodet{
    using namespace cv;
    using namespace std;

    static float desigmoid(float x){
        return -log(1.0f / x - 1.0f);
    }

    static float sigmoid(float x){
        return 1.0f / (1.0f + expf(-x));
    }

    static float iou(const kiwi::Box& a, const kiwi::Box& b){
        float cleft 	= max(a.left, b.left);
        float ctop 		= max(a.top, b.top);
        float cright 	= min(a.right, b.right);
        float cbottom 	= min(a.bottom, b.bottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
        float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
        return c_area / (a_area + b_area - c_area);
    }

    static void cpu_nms(kiwi::BoxArray& boxes, kiwi::BoxArray& output, float threshold){

        std::sort(boxes.begin(), boxes.end(), [](kiwi::BoxArray::const_reference a, kiwi::BoxArray::const_reference b){
            return a.confidence > b.confidence;
        });

        output.clear();
        vector<bool> remove_flags(boxes.size());
        for(int i = 0; i < boxes.size(); ++i){

            if(remove_flags[i]) continue;

            auto& a = boxes[i];
            output.emplace_back(a);

            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;
                
                auto& b = boxes[j];
                if(b.class_label == a.class_label){
                    if(iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
    }

    static void predict_to_bbox(float* ptr, int stride, float out[4]){

        float cache[8];
        for(int k = 0; k < 4; ++k, ptr += 8){
            float sum = 0;
            for(int i = 0; i < 8; ++i){
                cache[i] = expf(ptr[i]);
                sum += cache[i];
            }

            out[k] = 0;
            for(int i = 0; i < 8; ++i)
                out[k] += cache[i] / sum * i;

            out[k] *= stride;
        }
    }

    using ControllerImpl = kiwi::Producer
    <
        Mat,                    // input
        kiwi::BoxArray,         // output
        string                  // start param
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const std::string& engine_file,
            float confidence_threshold=0.25f, float nms_threshold=0.5f, bool no_sigmoid=false
        ){
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            no_sigmoid_           = no_sigmoid;

            if(no_sigmoid) confidence_threshold_ = desigmoid(confidence_threshold_);
            return ControllerImpl::startup(engine_file);
        }

        virtual void worker(promise<bool>& result) override{

            auto file = start_param_;
            auto mtype = kiwi::get_model_format(file);
            shared_ptr<kiwi::Infer> engine;
            kiwi::DPNN dpnn;

            if(mtype == kiwi::ModelType::RKNN){
                engine = rknn::load_infer(file);
            }else if(mtype == kiwi::ModelType::DPNN){
                if(!kiwi::load_dpnn(file, dpnn)){
                    result.set_value(false);
                    return;
                }

                if(dpnn.infer_type != "nanodet"){
                    INFOE("Model infer-type is not nanodet: %s [%s]", file.c_str(), dpnn.infer_type.c_str());
                    result.set_value(false);
                    return;
                }
                engine = rknn::load_infer_from_memory(dpnn.data.data(), dpnn.data.size());
                dpnn.data.clear();
            }else{
                INFOE("Invalid model type: %s", file.c_str());
                result.set_value(false);
                return;
            }

            if(engine == nullptr){
                result.set_value(false);
                INFOE("Load failed: %s", file.c_str());
                return;
            }

            if(mtype == kiwi::ModelType::DPNN){
                INFO("model: %s", dpnn.name.c_str());
                INFO("artist: %s", dpnn.artist.c_str());
                INFO("version: %s", dpnn.version.c_str());
                INFO("date_time: %s", dpnn.date_time.c_str());
                INFO("labels size: %d", dpnn.labels.size());
                for(auto item : dpnn.labels){
                    INFO("    %d - %s", item.first, item.second.c_str());
                }
            }
            engine->print();

            auto input         = engine->input();
            auto output        = engine->output();
            int num_classes    = output->size(2) - 32;
            int strides[]      = {8, 16, 32, 64};
            int nstride        = sizeof(strides) / sizeof(strides[0]);
            int linesize       = output->size(2);
            
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            vector<tuple<float, float, float>> grid;
            for(int i = 0; i < nstride; ++i){
                int stride = strides[i];
                int fw = (input_width_ + stride - 1) / stride;
                int fh = (input_height_ + stride - 1) / stride;
                for(int j = 0; j < fh; ++j){
                    for(int k = 0; k < fw; ++k)
                        grid.emplace_back(stride, k * stride, j * stride);
                }
            }

            result.set_value(true);

            Job fetch_job;
            kiwi::BoxArray output_objs, keep_output_objs;
            cv::Size target_size(input_width_, input_height_);
            cv::Mat input_image(input_height_, input_width_, CV_8UC3, input->cpu());
            float* optr = output->cpu<float>();
            std::vector<bool> remove_flags;
            int num_box = output->size(1);
            output_objs.reserve(100);
            keep_output_objs.reserve(100);
            remove_flags.resize(100);

            float sx = 0;
            float sy = 0;
            while(get_job_and_wait(fetch_job)){
                
                if(fetch_job.input.size() != target_size){
                    sx = fetch_job.input.cols / (float)target_size.width;
                    sy = fetch_job.input.rows / (float)target_size.height;
                    cv::resize(fetch_job.input, input_image, target_size, 0, 0, cv::InterpolationFlags::INTER_NEAREST);
                }else{
                    fetch_job.input.copyTo(input_image);
                    sx = 1;
                    sy = 1;
                }

                if(!engine->forward()){
                    fetch_job.pro->set_value({});
                    continue;
                }
                float* ptr = optr;
                for(int i = 0; i < num_box; ++i, ptr += linesize){
                    
                    int label = 0;
                    if(num_classes > 1){
                        label = std::max_element(ptr, ptr + num_classes) - ptr;
                    }

                    if(mtype == kiwi::ModelType::DPNN){
                        if(!dpnn.has_label(label))
                            continue;
                    }

                    float confidence = ptr[label];
                    if(confidence < confidence_threshold_) continue;

                    auto& g = grid[i];
                    float stride = get<0>(g);
                    float gx = get<1>(g);
                    float gy = get<2>(g);
                    float* pbox = ptr + num_classes;
                    float out[4];
                    predict_to_bbox(pbox, stride, out);

                    float left   = (gx - out[0]) * sx;
                    float top    = (gy - out[1]) * sy;
                    float right  = (gx + out[2]) * sx;
                    float bottom = (gy + out[3]) * sy;
                    output_objs.emplace_back(left, top, right, bottom, confidence, label);
                }
                
                cpu_nms(output_objs, keep_output_objs, nms_threshold_);
                if(no_sigmoid_){
                    for(auto& item : keep_output_objs)
                        item.confidence = sigmoid(item.confidence);
                }

                for(auto& item : keep_output_objs){
                    if(mtype == kiwi::ModelType::DPNN)
                        item.label_name = dpnn.label_to_name(item.class_label);
                    else{
                        char buf[100];
                        sprintf(buf, "%d", item.class_label);
                        item.label_name = buf;
                    }
                }
                fetch_job.pro->set_value(keep_output_objs);

                output_objs.clear();
                keep_output_objs.clear();
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{
            job.input = image;
            return !image.empty();
        }

        virtual std::shared_future<kiwi::BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        bool no_sigmoid_            = false;
    };

    shared_ptr<Infer> create_infer(
        const std::string& engine_file,
        float confidence_threshold, float nms_threshold, bool no_sigmoid
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, confidence_threshold, 
            nms_threshold, no_sigmoid)
        ){
            instance.reset();
        }
        return instance;
    }
};