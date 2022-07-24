#include "kiwi-app-arcface.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-infer-rknn.hpp"
#include "kiwi-logger.hpp"

namespace arcface{
    using namespace cv;
    using namespace std;

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        // float d2i[6];

        void compute(const float lands[10]){

            // 112 x 112分辨率时的标准人脸关键点（训练用的是这个）
            // 96  x 112分辨率时的标准人脸关键点在下面基础上去掉x的偏移
            // 来源于论文和公开代码中训练用到的
            // https://github.com/wy1iu/sphereface/blob/f5cd440a2233facf46b6529bd13231bb82f23177/preprocess/code/face_align_demo.m
            float Sdata[] = {
                30.2946 + 8, 51.6963,
                65.5318 + 8, 51.5014,
                48.0252 + 8, 71.7366,
                33.5493 + 8, 92.3655,
                62.7299 + 8, 92.2041
            };

            // 以下代码参考自：http://www.zifuture.com/archives/face-alignment
            float Qdata[] = {
                lands[0],  lands[1], 1, 0,
                lands[1], -lands[0], 0, 1,
                lands[2],  lands[3], 1, 0,
                lands[3], -lands[2], 0, 1,
                lands[4],  lands[5], 1, 0,
                lands[5], -lands[4], 0, 1,
                lands[6],  lands[7], 1, 0,
                lands[7], -lands[6], 0, 1,
                lands[8],  lands[9], 1, 0,
                lands[9], -lands[8], 0, 1,
            };
            
            float Udata[4];
            Mat_<float> Q(10, 4, Qdata);
            Mat_<float> U(4, 1,  Udata);
            Mat_<float> S(10, 1, Sdata);

            U = (Q.t() * Q).inv() * Q.t() * S;
            i2d[0] = Udata[0];   i2d[1] = Udata[1];     i2d[2] = Udata[2];
            i2d[3] = -Udata[1];  i2d[4] = Udata[0];     i2d[5] = Udata[3];

            // cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            // cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            // cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }
    };

    using ControllerImpl = kiwi::Producer
    <
        tuple<Mat, bool>,       // input
        Mat,                    // output
        string                  // start param
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

            auto engine = rknn::load_infer(start_param_);
            if(engine == nullptr){
                result.set_value(false);
                return;
            }
            start_param_.clear();
            engine->print();

            auto input         = engine->input();
            auto output        = engine->output();
            int feature_length = output->size(1);
            input_width_       = input->size(3);
            input_height_      = input->size(2);
            target_size_       = cv::Size(input_width_, input_height_);
            result.set_value(true);

            Job fetch_job;
            cv::Mat input_image(input_height_, input_width_, CV_8UC3, input->cpu());
            float* optr = output->cpu<float>();

            while(get_job_and_wait(fetch_job)){
                
                auto& image = get<0>(fetch_job.input);
                bool clone_output = get<1>(fetch_job.input);
                image.copyTo(input_image);
                if(!engine->forward()){
                    fetch_job.pro->set_value(cv::Mat());
                    continue;
                }

                auto feat = cv::Mat(1, feature_length, CV_32F, optr);
                cv::normalize(feat, feat, 1.0f, 0.0f, cv::NORM_L2, CV_32F);

                if(clone_output){
                    fetch_job.pro->set_value(feat.clone());
                }else{
                    fetch_job.pro->set_value(feat);
                }
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const tuple<Mat, bool>& input) override{
            
            auto& image = get<0>(input);
            if(image.size() != target_size_){
                INFOE("Invalid image size %d x %d, please use face_alignment instead, target_size is %d x %d", image.cols, image.rows, target_size_.width, target_size_.height);
                return false;
            }
            job.input = input;
            return true;
        }

        virtual std::shared_future<cv::Mat> commit(const Mat& image, bool clone_output) override{
            return ControllerImpl::commit(make_tuple(image, clone_output));
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

    cv::Mat face_alignment(const cv::Mat& image, const float face_landmark[10]){

        Size input_size(112, 112);
        AffineMatrix am;
        am.compute(face_landmark);

        Mat output;
        warpAffine(image, output, Mat_<float>(2, 3, am.i2d), input_size, cv::INTER_LINEAR);
        return output;
    }
};