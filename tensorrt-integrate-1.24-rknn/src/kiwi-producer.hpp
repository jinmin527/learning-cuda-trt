#ifndef KIWI_PRODUCER_HPP
#define KIWI_PRODUCER_HPP

#include <string>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include "kiwi-infer.hpp"

namespace kiwi{

    template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
    class Producer{
    public:
        struct Job{
            Input input;
            JobAdditional additional;
            std::shared_ptr<std::promise<Output>> pro;
        };

        virtual ~Producer(){
            stop();
        }

        void stop(){
            run_ = false;
            cond_.notify_all();

            ////////////////////////////////////////// cleanup jobs
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                while(!jobs_.empty()){
                    auto& item = jobs_.front();
                    if(item.pro)
                        item.pro->set_value(Output());
                    jobs_.pop();
                }
            };

            if(worker_){
                worker_->join();
                worker_.reset();
            }
        }

        bool startup(const StartParam& param){
            run_ = true;

            std::promise<bool> pro;
            start_param_ = param;
            worker_      = std::make_shared<std::thread>(&Producer::worker, this, std::ref(pro));
            return pro.get_future().get();
        }

        virtual std::shared_future<Output> commit(const Input& input){

            Job job;
            job.pro = std::make_shared<std::promise<Output>>();
            if(!preprocess(job, input)){
                job.pro->set_value(Output());
                return job.pro->get_future();
            }
            
            ///////////////////////////////////////////////////////////
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                jobs_.push(job);
            };
            cond_.notify_one();
            return job.pro->get_future();
        }

    protected:
        virtual void worker(std::promise<bool>& result) = 0;
        virtual bool preprocess(Job& job, const Input& input) = 0;
        
        virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size){

            std::unique_lock<std::mutex> l(jobs_lock_);
            cond_.wait(l, [&](){
                return !run_ || !jobs_.empty();
            });

            if(!run_) return false;
            
            fetch_jobs.clear();
            for(int i = 0; i < max_size && !jobs_.empty(); ++i){
                fetch_jobs.emplace_back(std::move(jobs_.front()));
                jobs_.pop();
            }
            return true;
        }

        virtual bool get_job_and_wait(Job& fetch_job){

            std::unique_lock<std::mutex> l(jobs_lock_);
            cond_.wait(l, [&](){
                return !run_ || !jobs_.empty();
            });

            if(!run_) return false;
            
            fetch_job = std::move(jobs_.front());
            jobs_.pop();
            return true;
        }

    protected:
        StartParam start_param_;
        std::atomic<bool> run_;
        std::mutex jobs_lock_;
        std::queue<Job> jobs_;
        std::shared_ptr<std::thread> worker_;
        std::condition_variable cond_;
    };
}; // kiwi

#endif // KIWI_PRODUCER_HPP