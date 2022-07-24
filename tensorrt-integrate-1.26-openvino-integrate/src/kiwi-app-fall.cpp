#include "kiwi-app-fall.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "kiwi-producer.hpp"
#include "kiwi-logger.hpp"

namespace fall{
    using namespace cv;
    using namespace std;

    const char* state_name(FallState state){
        switch(state){
            case FallState::Fall:      return "Fall";
            case FallState::Stand:     return "Stand";
            case FallState::UnCertain: return "UnCertain";
            default: return "Unknow";
        }
    }

    static void softmax(float* p, int size){

        float total = 0;
        for(int i = 0; i < size; ++i){
            p[i] = exp(p[i]);
            total += p[i];
        }

        for(int i = 0; i < size; ++i)
            p[i] /= total;
    }

    using ControllerImpl = kiwi::Producer
    <
        Input,                     // input
        tuple<FallState, float>,   // output
        string                     // start param
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

            // clean memory
            start_param_.clear();
            engine->print();

            auto input         = engine->input();
            auto output        = engine->output();
            result.set_value(true);

            Job fetch_job;
            float* optr = output->cpu<float>();
            while(get_job_and_wait(fetch_job)){

                auto& keys = get<0>(fetch_job.input);
                float* inptr = input->cpu<float>();
                for(int i = 0; i < keys.size(); ++i, inptr += 3){
                    auto& point = keys[i];
                    inptr[0] = point.x;
                    inptr[1] = point.y;
                    inptr[2] = point.z;
                }
                
                if(!engine->forward()){
                    INFOE("Forward failed");
                    fetch_job.pro->set_value({});
                    continue;
                }

                softmax(optr, output->size(1));
                int label = std::max_element(optr, optr + output->size(1)) - optr;
                fetch_job.pro->set_value(make_tuple((FallState)label, optr[label]));
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Input& input) override{

            if(get<0>(input).size() != 16){
                INFOE("keys.size()[%d] != 16", get<0>(input).size());
                return false;
            }

            job.input = input;
            auto& keys = get<0>(job.input);
            auto& box  = get<1>(job.input);
            int box_max_line = max(box.width, box.height);
            for(int i = 0; i < keys.size(); ++i){
                auto& point = keys[i];
                point.x = (point.x - box.x) / box_max_line - 0.5f;
                point.y = (point.y - box.y) / box_max_line - 0.5f;
            }
            return true;
        }

        virtual std::shared_future<std::tuple<FallState, float>> commit(const std::vector<cv::Point3f>& pose16, const cv::Rect& box) override{
            return ControllerImpl::commit(make_tuple(pose16, box));
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