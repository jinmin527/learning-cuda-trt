#include <thread>
#include <vector>
#include <future>
#include <chrono>

using namespace std;

void future_test(){

    promise<int> pro;

    // ref: https://stackoverflow.com/questions/21105169/is-there-any-difference-betwen-and-in-lambda-functions#:~:text=The%20difference%20is%20how%20the%20values%20are
    // shared_future可以被传参给别人使用，future不可以
    // 一旦pro获取得到future，则他们产生关联，future通过get等待pro的赋值
    shared_future<int> fut = pro.get_future();

    std::thread(
        [&](){ // lambda表达式 和 捕获列表[&]引用捕获方式 [=]值捕获方式
            printf("Async thread start.\n");

            this_thread::sleep_for(chrono::seconds(5));
            printf("Set value to 555.\n");
            pro.set_value(555);
            printf("Set value done.\n");
        }
    ).detach(); // join() 和 detach()

    printf("Wait value.\n");
    printf("fut.get() = %d\n", fut.get());
}