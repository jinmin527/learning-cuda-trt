#include <condition_variable>
#include <thread>
#include <vector>
#include <mutex>
#include <string>
#include <future>
#include <stdio.h>

using namespace std;

// 模仿写了一个条件变量
class MyConditionVariable{
public:
    template<typename _Lock, typename _Predict>
    void wait(
        _Lock& lock, const _Predict& p
    ){
        while(!p()){
            wait(lock);
        }
    }

    void notify_one(){
        has_notify_signal_ = true;
    }

private:
    // 真实场景是，wait可以多个线程同时wait，而signal则可以是数组，每个线程都可以消费掉一个信号
    template<typename _Lock>
    void wait(_Lock& lock){
        
        lock.unlock();
        while(!has_notify_signal_){
            this_thread::yield();
        }
        // 消费掉这个信号
        has_notify_signal_ = false;
        lock.lock();
    }

private:
    volatile bool has_notify_signal_ = false;
};

void condition_variable_test(){

    // 主要用于事件触发或等待，避免使用while sleep结构
    condition_variable cv;
    // MyConditionVariable cv; // 我们也可以自己实现一个condition variable类
    mutex lock_;
    atomic<bool> running_{true};

    auto func = [&](int tid){
        /*
        该函数主要干一件事情：
            running_ = false的时候且接收到信号的时候，打印一个done 
         */
        printf("Async thread start. tid = %d\n", tid);
        unique_lock<mutex> unique_lock_(lock_);
        
        printf("%d. Wait signal\n", tid);
        cv.wait(unique_lock_, [&](){
            printf("%d. 如果返回false，则继续等待，返回true退出等待\n", tid);
            return !running_; // !running_为true（就退出等待），!running_为false就继续等待
        });
        
        printf("%d. done.\n", tid); 
    }; // lambda表达式表示一个函数功能

    std::thread t0(func, 0);
    this_thread::sleep_for(chrono::seconds(3));
    printf("Notify one 1.\n");
    cv.notify_one();

    this_thread::sleep_for(chrono::seconds(3));
    running_ = false;
    printf("Notify one 2.\n"); //1
    cv.notify_one(); //2
    t0.join();
}