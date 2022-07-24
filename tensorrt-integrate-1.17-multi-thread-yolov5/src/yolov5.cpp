
#include "yolov5.hpp"
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <string>
#include <future>
#include <queue>
#include <functional>
#include "trt-infer.hpp"
#include "cuda-tools.hpp"
#include "simple-logger.hpp"

/////////////////////////////////////////////////////////////////////////////////////////
// 封装接口类
using namespace std;

namespace YoloV5{

    struct Job{
        shared_ptr<promise<BoxArray>> pro;
        cv::Mat input;
        float d2i[6];
    };

    class InferImpl : public Infer{
    public:
        virtual ~InferImpl(){
            stop();
        }

        void stop(){
            if(running_){
                running_ = false;
                cv_.notify_one();
            }

            if(worker_thread_.joinable())
                worker_thread_.join();
        }

        bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold){

            file_    = file;
            running_ = true;
            gpuid_   = gpuid;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;

            promise<bool> pro;
            worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));
            return pro.get_future().get();
        }

        virtual shared_future<BoxArray> commit(const cv::Mat& image) override{

            if(image.empty()){
                INFOE("Image is empty");
                return shared_future<BoxArray>();
            }
            
            Job job;
            job.pro.reset(new promise<BoxArray>());

            float scale_x = input_width_ / (float)image.cols;
            float scale_y = input_height_ / (float)image.rows;
            float scale   = std::min(scale_x, scale_y);
            float i2d[6];
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width_ + scale  - 1) * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height_ + scale - 1) * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, job.d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

            job.input.create(input_height_, input_width_, CV_8UC3);
            cv::warpAffine(image, job.input, m2x3_i2d, job.input.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
            job.input.convertTo(job.input, CV_32F, 1 / 255.0f);

            shared_future<BoxArray> fut = job.pro->get_future();
            {
                lock_guard<mutex> l(lock_);
                jobs_.emplace(std::move(job));
            }
            cv_.notify_one();
            return fut;
        }

        vector<Box> cpu_decode(
            float* predict, int rows, int cols, float* d2i,
            float confidence_threshold = 0.25f, float nms_threshold = 0.45f
        ){
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

                // 通过反变换恢复到图像尺度
                float left   = (cx - width * 0.5) * d2i[0] + d2i[2];
                float top    = (cy - height * 0.5) * d2i[0] + d2i[5];
                float right  = (cx + width * 0.5) * d2i[0] + d2i[2];
                float bottom = (cy + height * 0.5) * d2i[0] + d2i[5];
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
                    if(ibox.class_label == jbox.class_label){
                        // class matched
                        if(iou(ibox, jbox) >= nms_threshold)
                            remove_flags[j] = true;
                    }
                }
            }
            return box_result;
        }

        void worker(promise<bool>& pro){

            // load model
            checkRuntime(cudaSetDevice(gpuid_));
            auto model = TRT::load_infer(file_);

            if(model == nullptr){

                // failed
                pro.set_value(false);
                INFOE("Load model failed: %s", file_.c_str());
                return;
            }

            auto input    = model->input();
            auto output   = model->output();
            input_width_  = input->size(3);
            input_height_ = input->size(2);

            // load success
            pro.set_value(true);

            int max_batch_size = model->get_max_batch_size();
            vector<Job> fetched_jobs;
            while(running_){
                
                {
                    unique_lock<mutex> l(lock_);
                    cv_.wait(l, [&](){
                        return !running_ || !jobs_.empty();
                    });

                    if(!running_) break;
                    
                    for(int i = 0; i < max_batch_size && !jobs_.empty(); ++i){
                        fetched_jobs.emplace_back(std::move(jobs_.front()));
                        jobs_.pop();
                    }
                }

                for(int ibatch = 0; ibatch < fetched_jobs.size(); ++ibatch){
                    
                    auto& job = fetched_jobs[ibatch];
                    auto& image = job.input;
                    cv::Mat channel_based[3];
                    for(int i = 0; i < 3; ++i){
                        // 这里实现bgr -> rgb
                        // 做的是内存引用，效率最高
                        channel_based[i] = cv::Mat(input_height_, input_width_, CV_32F, input->cpu<float>(ibatch, 2-i));
                    }
                    cv::split(image, channel_based);
                }

                // 一次加载一批，并进行批处理
                // forward(fetched_jobs)
                model->forward();

                for(int ibatch = 0; ibatch < fetched_jobs.size(); ++ibatch){
                    auto& job = fetched_jobs[ibatch];
                    float* predict_batch = output->cpu<float>(ibatch);
                    auto boxes = cpu_decode(
                        predict_batch, output->size(1), output->size(2), job.d2i, confidence_threshold_, nms_threshold_
                    );
                    job.pro->set_value(boxes);
                }
                fetched_jobs.clear();
            }

            // 避免外面等待
            unique_lock<mutex> l(lock_);
            while(!jobs_.empty()){
                jobs_.back().pro->set_value({});
                jobs_.pop();
            }
            INFO("Infer worker done.");
        }

    private:
        atomic<bool> running_{false};
        int gpuid_;
        float confidence_threshold_;
        float nms_threshold_;
        int input_width_;
        int input_height_;
        string file_;
        thread worker_thread_;
        queue<Job> jobs_;
        mutex lock_;
        condition_variable cv_;
    };

    shared_ptr<Infer> create_infer(const string& file, int gpuid, float confidence_threshold, float nms_threshold){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(file, gpuid, confidence_threshold, nms_threshold)){
            instance.reset();
        }
        return instance;
    }
};