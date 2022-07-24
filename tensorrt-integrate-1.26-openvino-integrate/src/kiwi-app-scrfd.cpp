#include "kiwi-app-scrfd.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-logger.hpp"

namespace scrfd{
    using namespace cv;
    using namespace std;
    

    static float desigmoid(float x){
        return -log(1.0f / x - 1.0f);
    }

    static float sigmoid(float x){
        return 1.0f / (1.0f + expf(-x));
    }

    static float iou(const kiwi::Face& a, const kiwi::Face& b){
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

    static void cpu_nms(kiwi::FaceArray& boxes, kiwi::FaceArray& output, float threshold){

        std::sort(boxes.begin(), boxes.end(), [](kiwi::FaceArray::const_reference a, kiwi::FaceArray::const_reference b){
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
                if(iou(a, b) >= threshold)
                    remove_flags[j] = true;
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
        kiwi::FaceArray,        // output
        string,                  // start param
        tuple<float, float>     // sx, sy
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const std::string& engine_file,
            float confidence_threshold=0.25f, float nms_threshold=0.5f
        ){
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            return ControllerImpl::startup(engine_file);
        }

        size_t compute_prior_size(int input_width, int input_height, const vector<int>& strides={8, 16, 32}, int num_anchor_per_stage=2){

            size_t total = 0;
            for(int s : strides){
                int feature_map_width  = (input_width + s - 1)  / s;
                int feature_map_height = (input_height + s - 1) / s;
                total += feature_map_width * feature_map_height * num_anchor_per_stage;
            }
            return total;
        }

        void init_prior_box(vector<tuple<float, float, float, float>>& prior, int input_width, int input_height){

            vector<int> strides{8, 16, 32};
            vector<vector<float>> min_sizes{
                vector<float>({16.0f,  32.0f }),
                vector<float>({64.0f,  128.0f}),
                vector<float>({256.0f, 512.0f})
            };
            prior.resize(compute_prior_size(input_width, input_height, strides));
            
            int prior_row = 0;
            for(int istride = 0; istride < strides.size(); ++istride){
                int stride         = strides[istride];
                auto anchor_sizes  = min_sizes[istride];
                int feature_map_width  = (input_width + stride - 1)  / stride;
                int feature_map_height = (input_height + stride - 1) / stride;
                
                for(int y = 0; y < feature_map_height; ++y){
                    for(int x = 0; x < feature_map_width; ++x){
                        for(int isize = 0; isize < anchor_sizes.size(); ++isize){
                            float anchor_size = anchor_sizes[isize];
                            float dense_cx    = x * stride;
                            float dense_cy    = y * stride;
                            float s_kx        = stride;
                            float s_ky        = stride;
                            prior[prior_row++] = make_tuple(dense_cx, dense_cy, s_kx, s_ky);
                        }
                    }
                }
            }
        }

        virtual void worker(promise<bool>& result) override{

            auto engine = kiwi::load_infer(start_param_);
            if(engine == nullptr){
                result.set_value(false);
                return;
            }

            // clean memory
            start_param_.clear();
            engine->print();

            auto input         = engine->input();
            auto output        = engine->output();
            int linesize       = output->size(2);
            int num_box        = output->size(1);
            float deconfidence_threshold = desigmoid(confidence_threshold_);
            
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            vector<tuple<float, float, float, float>> prior;
            init_prior_box(prior, input_width_, input_height_);
            if(prior.size() != num_box){
                INFOE("Invalid model, prior.size[%d] != num_box[%d]", prior.size(), num_box);
                result.set_value(false);
                return;
            }
            result.set_value(true);

            Job fetch_job;
            kiwi::FaceArray output_objs, keep_output_objs;
            target_size_ = cv::Size(input_width_, input_height_);

            cv::Mat channel_based[3];
            for(int i = 0; i  < 3; ++i)
                channel_based[i] = cv::Mat(input_height_, input_width_, CV_32F, input->cpu<float>(0, i));

            float* optr = output->cpu<float>();
            std::vector<bool> remove_flags;
            output_objs.reserve(100);
            keep_output_objs.reserve(100);
            remove_flags.resize(100);

            while(get_job_and_wait(fetch_job)){
                
                cv::split(fetch_job.input, channel_based);
                if(!engine->forward()){
                    INFOE("Forward failed");
                    fetch_job.pro->set_value({});
                    continue;
                }

                float sx = get<0>(fetch_job.additional);
                float sy = get<1>(fetch_job.additional);
                float* ptr = optr;
                for(int i = 0; i < num_box; ++i, ptr += linesize){
                    
                    //cx, cy, w, h, conf, landmark0.x, landmark0.y, landmark1.x, landmark1.y, landmark2.x, landmark2.y
                    float confidence = ptr[4];
                    if(confidence < deconfidence_threshold) continue;

                    auto& p = prior[i];
                    float dense_cx = get<0>(p);
                    float dense_cy = get<1>(p);
                    float s_kx = get<2>(p);
                    float s_ky = get<3>(p);
                    float dx      = ptr[0] * s_kx;
                    float dy      = ptr[1] * s_ky;
                    float dr      = ptr[2] * s_kx;
                    float db      = ptr[3] * s_ky;
                    output_objs.emplace_back();

                    kiwi::Face& face = output_objs.back();
                    face.left    = (dense_cx - dx) * sx;
                    face.top     = (dense_cy - dy) * sy;
                    face.right   = (dense_cx + dr) * sx;
                    face.bottom  = (dense_cy + db) * sy;
                    face.confidence = sigmoid(confidence);

                    float* landmark_predict = ptr + 5;
                    float* plandmark_out = face.landmark;
                    for(int i = 0; i < 5; ++i){
                        plandmark_out[0] = (dense_cx + landmark_predict[0] * s_kx) * sx;
                        plandmark_out[1] = (dense_cy + landmark_predict[1] * s_ky) * sy;
                        landmark_predict += 2;
                        plandmark_out += 2;
                    }
                }
                
                cpu_nms(output_objs, keep_output_objs, nms_threshold_);
                fetch_job.pro->set_value(keep_output_objs);

                output_objs.clear();
                keep_output_objs.clear();
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{

            if(image.empty()){
                INFOE("image is empty");
                return false;
            }

            float sx, sy;
            if(image.size() != target_size_){
                sx = image.cols / (float)target_size_.width;
                sy = image.rows / (float)target_size_.height;
                cv::resize(image, job.input, target_size_, 0, 0, cv::InterpolationFlags::INTER_LINEAR);
            }else{
                image.copyTo(job.input);
                sx = 1;
                sy = 1;
            }
            job.additional = make_tuple(sx, sy);
            job.input.convertTo(job.input, CV_32F, 1/127.5, -1.0f);
            return true;
        }

        virtual std::shared_future<kiwi::FaceArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        cv::Size target_size_;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
    };

    shared_ptr<Infer> create_infer(
        const std::string& engine_file,
        float confidence_threshold, float nms_threshold
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, confidence_threshold, 
            nms_threshold)
        ){
            instance.reset();
        }
        return instance;
    }
};