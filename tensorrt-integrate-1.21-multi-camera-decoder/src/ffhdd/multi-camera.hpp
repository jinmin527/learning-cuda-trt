#ifndef MULTI_CAMERA_HPP
#define MULTI_CAMERA_HPP

#include <functional>
#include <memory>
#include <string>
#include "cuvid-decoder.hpp"
#include "ffmpeg-demuxer.hpp"

namespace FFHDMultiCamera{

    class View;
    typedef std::function<void(
        View* pview,
        uint8_t* pimage_data, int device_id, int width, int height, 
        FFHDDecoder::FrameType type, uint64_t timestamp, 
        FFHDDecoder::ICUStream stream)
    > decode_callback;

    class View{
    public:
        virtual void set_callback(decode_callback callback) = 0;
        virtual std::shared_ptr<FFHDDemuxer::FFmpegDemuxer> get_demuxer() = 0;
        virtual bool demux(bool only_push_keyframe = false) = 0;
        virtual int get_idd() = 0;
    };

    class Decoder{
    public:
        virtual std::shared_ptr<View> make_view(const std::string& uri, bool auto_reboot=false) = 0;
        virtual void join() = 0;
    };

    std::shared_ptr<Decoder> create_decoder(
        bool use_device_frame=true, int max_cache = -1, int gpu_id = -1, 
        const FFHDDecoder::CropRect *crop_rect = nullptr, 
        const FFHDDecoder::ResizeDim *resize_dim = nullptr, 
        FFHDDecoder::FrameType type = FFHDDecoder::FrameType::BGR
    );
} //FFHDMultiCamera


#endif // MULTI_CAMERA_HPP