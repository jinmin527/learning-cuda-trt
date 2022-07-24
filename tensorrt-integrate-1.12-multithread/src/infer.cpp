
#include "infer.hpp"
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <string>
#include <future>
#include <queue>
#include <functional>

// 封装接口类
using namespace std;

struct Job{
    shared_ptr<promise<string>> pro;
    string input;
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

    bool startup(const string& file){

        file_ = file;
        running_ = true; // 启动后，运行状态设置为true

        // 线程传递promise的目的，是获得线程是否初始化成功的状态
        // 而在线程内做初始化，好处是，初始化跟释放在同一个线程内
        // 代码可读性好，资源管理方便
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));
        /* 
            注意：这里thread 一构建好后，worker函数就开始执行了
            第一个参数是该线程要执行的worker函数，第二个参数是this指的是class InferImpl，第三个参数指的是传引用，因为我们在worker函数里要修改pro。
         */
        return pro.get_future().get();
    }

    virtual shared_future<string> commit(const string& input) override{
        /* 
        建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            commit 函数 https://v.douyin.com/NfJvHxm/
         */
        Job job;
        job.input = input;
        job.pro.reset(new promise<string>());

        shared_future<string> fut = job.pro->get_future();
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));
        }
        cv_.notify_one();
        return fut;
    }

    void worker(promise<bool>& pro){
        /* 
        建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            worker函数 https://v.douyin.com/NfJPojm/
         */

        // load model
        if(file_ != "trtfile"){

            // failed
            pro.set_value(false);
            printf("Load model failed: %s\n", file_.c_str());
            return;
        }

        // load success
        pro.set_value(true); // 这里的promise用来负责确认infer初始化成功了

        vector<Job> fetched_jobs;
        while(running_){
            
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();}); // 一直等着，cv_.wait(lock, predicate) // 如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号

                if(!running_) break; // 如果 不在运行 就直接结束循环
                
                int batch_size = 5;
                for(int i = 0; i < batch_size && !jobs_.empty(); ++i){   // jobs_不为空的时候
                    fetched_jobs.emplace_back(std::move(jobs_.front())); // 就往里面fetched_jobs里塞东西
                    jobs_.pop();                                         // fetched_jobs塞进来一个，jobs_那边就要pop掉一个。（因为move）
                }
            }

            // 一次加载一批，并进行批处理
            // forward(fetched_jobs)
            for(auto& job : fetched_jobs){
                job.pro->set_value(job.input + "---processed");
            }
            fetched_jobs.clear();
        }
        printf("Infer worker done.\n");
    }

private:
    atomic<bool> running_{false};
    string file_;
    thread worker_thread_;
    queue<Job> jobs_;
    mutex lock_;
    condition_variable cv_;
};

shared_ptr<Infer> create_infer(const string& file){
    /* 
        [建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)]
        RAII+封装接口模式：问题定义-异常流程处理 https://v.douyin.com/NfJtnpF/
        RAII+封装接口模式：解决方案-用户友好设计 https://v.douyin.com/NfJteyc/
     */
    shared_ptr<InferImpl> instance(new InferImpl()); // 实例化一个推理器的实现类（inferImpl），以指针形式返回 
    if(!instance->startup(file)){                    // 推理器实现类实例(instance)启动。这里的file是engine file
        instance.reset();                            // 如果启动不成功就reset
    }
    return instance;    
}

void infer_test(){
    auto infer = create_infer("trtfile"); // 创建及初始化 抖音网页短视频辅助讲解: 创建及初始化推理器 https://v.douyin.com/NfJvWdW/
    if(infer == nullptr){                       
        printf("Infer is nullptr.\n");          
        return;
    }

    printf("commit msg = %s\n", infer->commit("msg").get().c_str()); // 将任务提交给推理器（推理器执行commit），同时推理器（infer）也等着获取（get）结果
}