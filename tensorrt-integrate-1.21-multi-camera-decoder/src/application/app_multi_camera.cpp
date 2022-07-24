
#include <opencv2/opencv.hpp>
#include <ffhdd/simple-logger.hpp>
#include <ffhdd/ffmpeg-demuxer.hpp>
#include <ffhdd/cuvid-decoder.hpp>
#include <ffhdd/nalu.hpp>
#include <queue>
#include <mutex>
#include <tuple>
#include <future>
#include <condition_variable>
#include <ffhdd/cuda-tools.hpp>
#include <ffhdd/multi-camera.hpp>

using namespace std;

static void test_hard_decode(){

    auto decoder = FFHDMultiCamera::create_decoder(false, -1, 0);
    vector<thread> ts;
    int ids[64] = {0};
    auto callback = [&](
        FFHDMultiCamera::View* pview,
        uint8_t* pimage_data, int device_id, int width, int height, 
        FFHDDecoder::FrameType type, uint64_t timestamp, 
        FFHDDecoder::ICUStream stream
    ){
        cv::Mat image(height, width, CV_8UC3, pimage_data);
        cv::imwrite(cv::format("imgs/%02d_%03d.jpg", pview->get_idd(), ++ids[pview->get_idd()]), image);
        std::cout << "get_width:" << width<<",get_height:"<<height<<std::endl;
    };

    auto func = [&](shared_ptr<FFHDMultiCamera::View> view){
        if(view == nullptr){
            INFOE("View is nullptr");
            return;
        }

        view->set_callback(callback);
        while(view->demux()){
            // 模拟真实视频流
            this_thread::sleep_for(chrono::milliseconds(30));
        }
        INFO("Done> %d", view->get_idd());
    };

    for(int i = 0; i < 64; ++i){
        if(i % 3 == 0)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/dog.mp4")));
        else if(i % 3 == 1)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/cat.mp4")));
        else if(i % 3 == 2)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/pig.mp4")));
    }
    for(auto& t : ts)
        t.join();

    decoder->join();
    INFO("Program done.");
}

int app_multi_camera(){

    test_hard_decode();
    return 0;
}