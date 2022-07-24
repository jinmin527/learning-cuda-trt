#include "kiwi-app-yolov5.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-logger.hpp"

namespace yolov5{
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

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            // 这里取min的理由是
            // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            // **
            float scale = std::min(scale_x, scale_y);

            /**
            这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
            scale, 0, -scale * from.width * 0.5 + to.width * 0.5
            0, scale, -scale * from.height * 0.5 + to.height * 0.5
            
            这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
            例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
            S = [
            scale,     0,      0
            0,     scale,      0
            0,         0,      1
            ]
            
            P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
            P = [
            1,        0,      -scale * from.width * 0.5
            0,        1,      -scale * from.height * 0.5
            0,        0,                1
            ]
            T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
            T = [
            1,        0,      to.width * 0.5,
            0,        1,      to.height * 0.5,
            0,        0,            1
            ]
            通过将3个矩阵顺序乘起来，即可得到下面的表达式：
            M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
            ]
            去掉第三行就得到opencv需要的输入2x3矩阵
            **/

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = kiwi::Producer
    <
        Mat,                    // input
        kiwi::BoxArray,        // output
        string,                  // start param
        AffineMatrix     // sx, sy
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
            auto output        = engine->output(0);
            int output_numbox  = output->size(1);
            int output_numprob = output->size(2);
            int num_classes = output_numprob - 5;
            float* output_data_host = output->cpu<float>();
            
            input_width_       = input->size(3);
            input_height_      = input->size(2);
            
            result.set_value(true);

            Job fetch_job;
            kiwi::BoxArray output_objs, keep_output_objs;
            target_size_ = cv::Size(input_width_, input_height_);

            float* optr = output->cpu<float>();
            output_objs.reserve(100);
            keep_output_objs.reserve(100);

            while(get_job_and_wait(fetch_job)){

                memcpy(input->cpu<float>(), fetch_job.input.data, target_size_.area() * 3 * sizeof(float));
                if(!engine->forward()){
                    INFOE("Forward failed");
                    fetch_job.pro->set_value({});
                    continue;
                } 

                float* d2i = fetch_job.additional.d2i;
                for(int i = 0; i < output_numbox; ++i){
                    float* ptr = output_data_host + i * output_numprob;
                    float objness = ptr[4];
                    if(objness < confidence_threshold_)
                        continue;

                    float* pclass = ptr + 5;
                    int label     = std::max_element(pclass, pclass + num_classes) - pclass;
                    float prob    = pclass[label];
                    float confidence = prob * objness;
                    if(confidence < confidence_threshold_)
                        continue;

                    // 中心点、宽、高
                    float cx     = ptr[0];
                    float cy     = ptr[1];
                    float width  = ptr[2];
                    float height = ptr[3];

                    // 预测框
                    float left   = cx - width * 0.5;
                    float top    = cy - height * 0.5;
                    float right  = cx + width * 0.5;
                    float bottom = cy + height * 0.5;

                    // 对应图上的位置
                    float image_base_left   = d2i[0] * left   + d2i[2];
                    float image_base_right  = d2i[0] * right  + d2i[2];
                    float image_base_top    = d2i[0] * top    + d2i[5];
                    float image_base_bottom = d2i[0] * bottom + d2i[5];
                    output_objs.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, confidence, (float)label});
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

            cv::Mat temp_image;
            job.additional.compute(image.size(), target_size_);
            cv::warpAffine(image, temp_image, job.additional.i2d_mat(), target_size_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114.0f));

            job.input.create(target_size_, CV_32FC3);
            int image_area = target_size_.width * target_size_.height;
            unsigned char* pimage = temp_image.data;
            float* input_data_host = job.input.ptr<float>(0);
            float* phost_b = input_data_host + image_area * 0;
            float* phost_g = input_data_host + image_area * 1;
            float* phost_r = input_data_host + image_area * 2;
            for(int i = 0; i < image_area; ++i, pimage += 3){
                // 注意这里的顺序rgb调换了
                *phost_r++ = pimage[0] / 255.0f;
                *phost_g++ = pimage[1] / 255.0f;
                *phost_b++ = pimage[2] / 255.0f;
            }
            return true;
        }

        virtual std::shared_future<kiwi::BoxArray> commit(const Mat& image) override{
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