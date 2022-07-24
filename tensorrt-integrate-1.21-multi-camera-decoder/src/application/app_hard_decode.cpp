
#include <opencv2/opencv.hpp>
#include <ffhdd/simple-logger.hpp>
#include <ffhdd/ffmpeg-demuxer.hpp>
#include <ffhdd/cuvid-decoder.hpp>
#include <ffhdd/nalu.hpp>

using namespace std;

static void test_hard_decode(){

    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("exp/number100.mp4");
    if(demuxer == nullptr){
        INFOE("demuxer create failed");
        return;
    }

    auto decoder = FFHDDecoder::create_cuvid_decoder(
        false, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0
    );

    if(decoder == nullptr){
        INFOE("decoder create failed");
        return;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    uint64_t demuxpts = 0;
    uint64_t pts;

    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);

    int i = 0;
    do{
        i ++;

        demuxer->demux(&packet_data, &packet_size, &demuxpts);
        if(i % 3 == 0){
            demuxpts = (0x01ul << 60) | demuxpts;
        }else if(i % 3 == 1){
            demuxpts = (0x02ul << 60) | demuxpts;
        }else if(i % 3 == 2){
            demuxpts = (0x03ul << 60) | demuxpts;
        }
        
        if(packet_size == 0) demuxpts = 0;
        INFO("demux : %ld", demuxpts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, demuxpts);
        for(int i = 0; i < ndecoded_frame; ++i){
            unsigned int frame_index = 0;

            /* 因为decoder获取的frame内存，是YUV-NV12格式的。储存内存大小是 [height * 1.5] * width byte
             因此构造一个height * 1.5,  width 大小的空间
             然后由opencv函数，把YUV-NV12转换到BGR，转换后的image则是正常的height, width, CV_8UC3
            */
            // cv::Mat image(decoder->get_height() * 1.5, decoder->get_width(), CV_8U, decoder->get_frame(&pts, &frame_index));
            // cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);
            decoder->get_frame(&pts, &frame_index);
            // frame_index = frame_index + 1;
            // INFO("write imgs/img_%05d.jpg  %dx%d", frame_index, image.cols, image.rows);
            // cv::imwrite(cv::format("imgs/img_%05d.jpg", frame_index), image);
            INFO("%d, %ld", frame_index + 1, pts);
        }
    }while(packet_size > 0);
}

int app_hard_decode(){

    test_hard_decode();
    return 0;
}