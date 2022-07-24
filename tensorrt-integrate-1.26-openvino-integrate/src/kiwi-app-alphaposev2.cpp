#include "kiwi-app-alphaposev2.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-logger.hpp"

namespace alphaposev2{
    using namespace cv;
    using namespace std;

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& image_size, const cv::Rect& box, const cv::Size& net_size){
            Rect box_ = box;
            if(box_.width == 0 || box_.height == 0){
                box_.width  = image_size.width;
                box_.height = image_size.height;
                box_.x = 0;
                box_.y = 0;
            }

            float rate = box_.width > 100 ? 0.1f : 0.15f;
            float pad_width  = box_.width  * (1 + 2 * rate);
            float pad_height = box_.height * (1 + 1 * rate);
            float scale = min(net_size.width  / pad_width,  net_size.height / pad_height);
            i2d[0] = scale;  i2d[1] = 0;      i2d[2] = -(box_.x - box_.width  * 1 * rate + pad_width * 0.5)  * scale + net_size.width  * 0.5;  
            i2d[3] = 0;      i2d[4] = scale;  i2d[5] = -(box_.y - box_.height * 1 * rate + pad_height * 0.5) * scale + net_size.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    static tuple<float, float> affine_project(float x, float y, float* pmatrix){

        float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
        float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
        return make_tuple(newx, newy);
    }

    using ControllerImpl = kiwi::Producer
    <
        tuple<Mat, Rect>,         // input
        std::vector<cv::Point3f>, // output
        string,                   // start param
        AffineMatrix
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const std::string& engine_file
        ){
            return ControllerImpl::startup(engine_file);
        }

        virtual void worker(promise<bool>& result) override{

            auto engine = kiwi::load_infer(start_param_);
            if(engine == nullptr){
                result.set_value(false);
                return;
            }
            start_param_.clear();
            engine->print();

            auto input         = engine->input();
            auto output        = engine->output();
            input_width_       = input->size(3);
            input_height_      = input->size(2);
            target_size_       = cv::Size(input_width_, input_height_);
            result.set_value(true);

            Job fetch_job;
            cv::Mat channel_based[3];
            for(int i = 0; i  < 3; ++i)
                channel_based[i] = cv::Mat(input_height_, input_width_, CV_32F, input->cpu<float>(0, i));

            int begin_channel = 17;
            int area = output->size(2) * output->size(3);
            int stride        = input->size(3) / output->size(3);
            vector<Point3f> image_based_keypoints(output->size(1) - begin_channel);

            while(get_job_and_wait(fetch_job)){
                
                auto& image = get<0>(fetch_job.input);
                cv::split(image, channel_based);

                if(!engine->forward()){
                    fetch_job.pro->set_value(cv::Mat());
                    continue;
                }
                
                for(int i = begin_channel; i < output->size(1); ++i){
                    float* output_channel = output->cpu<float>(0, i);
                    int location = std::max_element(output_channel, output_channel + area) - output_channel;
                    float confidence = output_channel[location];
                    float x = (location % output->size(3)) * stride;
                    float y = (location / output->size(3)) * stride;
                    auto& output_point = image_based_keypoints[i-begin_channel];

                    output_point.z = confidence;
                    tie(output_point.x, output_point.y) = affine_project(x, y, fetch_job.additional.d2i);
                }
                fetch_job.pro->set_value(image_based_keypoints);
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const tuple<Mat, Rect>& input) override{
            
            auto& image = get<0>(input);
            if(image.empty()){
                INFOE("image is empty.");
                return false;
            }

            auto& output = get<0>(job.input);
            auto& box = get<1>(input);
            job.additional.compute(image.size(), box, target_size_);
            cv::warpAffine(image, output, job.additional.i2d_mat(), target_size_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(127.0f));
            cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
            output.convertTo(output, CV_32F, 1/255.0f);
            output -= cv::Scalar(0.406, 0.457, 0.480);
            return true;
        }

        virtual std::shared_future<vector<Point3f>> commit(const Mat& image, cv::Rect box) override{
            return ControllerImpl::commit(make_tuple(image, box));
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        cv::Size target_size_;
    };

    shared_ptr<Infer> create_infer(
        const std::string& engine_file
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file)
        ){
            instance.reset();
        }
        return instance;
    }
};