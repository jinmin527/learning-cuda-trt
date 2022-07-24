
#include "multi-camera.hpp"
#include <vector>
#include <queue>
#include <mutex>
#include <tuple>
#include <thread>
#include <future>
#include <list>
#include <condition_variable>
#include <string.h>
#include <ffhdd/nalu.hpp>
#include "cuda-tools.hpp"

namespace FFHDMultiCamera{

    using namespace std;

    struct Packet{
        vector<uint8_t> data_;
        uint64_t timestamp_ = 0;
        bool iskey_ = false;
        int idd_ = 0;

        Packet() = default;
        Packet(int idd, const uint8_t* pdata, size_t size, bool iskey, uint64_t timestamp){
            data_.resize(size);
            memcpy(data_.data(), pdata, size);
            timestamp_ = timestamp;
            iskey_ = iskey;
            idd_ = idd;
        }
    };

    class DecoderImpl;
    class ViewImpl : public View{
    public:
        ViewImpl(DecoderImpl* multi_camera_decoder, shared_ptr<FFHDDemuxer::FFmpegDemuxer> demuxer, int idd){
            decoder_ = multi_camera_decoder;
            demuxer_ = demuxer;
            idd_ = idd;
        }

        virtual void set_callback(decode_callback callback) override{callback_ = callback;}

        virtual void do_new_frame(
            uint8_t* pimage_data, int device_id, int width, int height, 
            FFHDDecoder::FrameType type, uint64_t timestamp, FFHDDecoder::ICUStream stream
        ){
            if(callback_){
                callback_(this, pimage_data, device_id, width, height, type, timestamp, stream);
            }
        }

        virtual shared_ptr<FFHDDemuxer::FFmpegDemuxer> get_demuxer()override{return demuxer_;}
        virtual bool demux(bool only_push_keyframe = false) override;
        virtual int get_idd() override{return idd_;}

    private:
        decode_callback callback_;
        shared_ptr<FFHDDemuxer::FFmpegDemuxer> demuxer_;
        DecoderImpl* decoder_ = nullptr;
        int idd_ = 0;
    };

    class DecoderImpl : public Decoder{
    public:
        DecoderImpl(
            bool use_device_frame, int max_cache = -1, int gpu_id = -1, 
            const FFHDDecoder::CropRect *crop_rect = nullptr, 
            const FFHDDecoder::ResizeDim *resize_dim = nullptr, 
            FFHDDecoder::FrameType type = FFHDDecoder::FrameType::BGR
        ){
            use_device_frame_ = use_device_frame;
            max_cache_ = max_cache;
            gpu_id_ = gpu_id;
            type_ = type;
            if(crop_rect) crop_rect_ = *crop_rect;
            if(resize_dim) resize_dim_ = *resize_dim;
        }

        virtual ~DecoderImpl(){
            
            running_ = false;
            cv_.notify_one();
            if(worker_thread_.joinable())
                worker_thread_.join();
        }

        virtual shared_ptr<View> make_view(const std::string& uri, bool auto_reboot=false) override{

            if(views_.size() >= max_idd_){
                INFOE("More than %d views are not supported", max_idd_);
                return nullptr;
            }

            auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, auto_reboot);
            if(demuxer == nullptr)
                return nullptr;

            std::lock_guard<mutex> l(make_view_lock_);
            if(!running_){
                
                auto codec = demuxer->get_video_codec();
                running_ = true;

                promise<bool> pro;
                uint8_t* packet_data = nullptr;
                int packet_size = 0;

                demuxer->get_extra_data(&packet_data, &packet_size);
                worker_thread_ = thread(&DecoderImpl::worker, this, 
                    codec, 
                    packet_data, packet_size,
                    std::ref(pro)
                );
                if(!pro.get_future().get()){
                    running_ = false;
                    worker_thread_.join();
                    return nullptr;
                }

                demuxer_width_ = demuxer->get_width();
                demuxer_height_ = demuxer->get_height();
            }

            if(demuxer_width_ != demuxer->get_width() || demuxer_height_ != demuxer->get_height()){
                INFOE(
                    "Demuxer must be same size, %d x %d != %d x %d: %s",
                    demuxer_width_, demuxer->get_width(), demuxer_height_, demuxer->get_height(),
                    uri.c_str()
                );
                return nullptr;
            }

            auto view = make_shared<ViewImpl>(this, demuxer, views_.size());
            views_.emplace_back(view);
            return view;
        }

        void push_demux(Packet&& packet){
            
            std::lock_guard<mutex> l(packet_lock_);
            packets_.push_back(std::move(packet));
            cv_.notify_one();
        }

        virtual void join() override{
            
            wait_prop_finish_ = true;
            cv_.notify_one();
            if(worker_thread_.joinable())
                worker_thread_.join();
            
            running_ = false;
        }

        void worker(FFHDDemuxer::IAVCodecID codec, uint8_t* pdata, size_t size, promise<bool>& pro){

            auto decoder = FFHDDecoder::create_cuvid_decoder(
                use_device_frame_, FFHDDecoder::ffmpeg2NvCodecId(codec), max_cache_,
                gpu_id_, &crop_rect_, &resize_dim_, type_
            );
            if(decoder == nullptr){
                pro.set_value(false);
                return;
            }
            decoder->decode(pdata, size, 0);
            pro.set_value(true);

            int device_id = use_device_frame_ ? gpu_id_ : -1;
            uint64_t mask_timestamp = ~((uint64_t)0xFFFFFFFF << num_bit_timestamp_);
            uint64_t pts = 0;
            uint64_t base_pts = 0;
            int current_idd = -1;
            list<Packet> local_packets;
            Packet current_packet;
            bool prev_find_item = true;
            bool first = true;
            while(running_){
                {
                    unique_lock<mutex> l(packet_lock_);

                    // return true则不等待，return false则等待
                    cv_.wait(l, [&](){
                        return prev_find_item && !local_packets.empty() || !packets_.empty() || !running_ || wait_prop_finish_;
                    });

                    if(!running_) break;
                    while(!packets_.empty()){
                        local_packets.emplace_back(std::move(packets_.front()));
                        packets_.pop_front();
                    }
                }

                if(local_packets.empty()){ 
                    if(wait_prop_finish_)
                        break;
                    continue;
                }

                if(current_idd == -1){
                    current_packet = std::move(local_packets.front());
                    local_packets.pop_front();
                    current_idd = current_packet.idd_;
                }else{
                    auto iter = local_packets.begin();
                    prev_find_item = false;
                    for(; iter != local_packets.end(); ++iter){
                        if(iter->idd_ == current_idd){
                            prev_find_item = true;
                            if(iter->iskey_){
                                current_packet = std::move(local_packets.front());
                                local_packets.pop_front();
                                current_idd = current_packet.idd_;
                            }else{
                                current_packet = std::move(*iter);
                                local_packets.erase(iter);
                            }
                            break;
                        }
                    }
                    if(!prev_find_item){
                        if(wait_prop_finish_){
                            break;
                        }
                        continue;
                    }
                }

                if(first){
                    first = false;
                    base_pts = current_packet.timestamp_;
                }

                int nframes = decoder->decode(
                    current_packet.data_.data(), 
                    current_packet.data_.size(), 
                    ((uint64_t)current_packet.idd_ << num_bit_timestamp_) | ((current_packet.timestamp_ - base_pts) & mask_timestamp)
                );

                std::lock_guard<mutex> l(make_view_lock_);
                for(int i = 0; i < nframes; ++i){

                    auto ptr = decoder->get_frame(&pts);
                    int view_id = pts >> num_bit_timestamp_;
                    auto new_pts = ((pts << num_bit_idd_) >> num_bit_idd_) + base_pts;
                    if(view_id >= 0 && view_id < views_.size()){
                        views_[view_id]->do_new_frame(
                            ptr, device_id, decoder->get_width(), decoder->get_height(), type_, new_pts, decoder->get_stream()
                        );
                    }else{
                        INFOW("View id[%d] out of range", view_id);
                    }
                }
            }
            INFO("Thread done.");
        }   

    private:
        int demuxer_width_ = 0;
        int demuxer_height_ = 0;
        atomic<bool> running_{false};
        atomic<bool> wait_prop_finish_{false};
        thread worker_thread_;
        mutex packet_lock_;
        mutex make_view_lock_;
        condition_variable cv_;
        deque<Packet> packets_;
        vector<shared_ptr<ViewImpl>> views_;
        bool use_device_frame_ = false;
        int max_cache_ = -1;
        int gpu_id_ = -1;
        FFHDDecoder::CropRect crop_rect_ = {0}; 
        FFHDDecoder::ResizeDim resize_dim_ = {0}; 
        FFHDDecoder::FrameType type_ = FFHDDecoder::FrameType::BGR;
        const int num_bit_idd_ = 6;
        const int num_bit_timestamp_ = 64 - num_bit_idd_;
        const int max_idd_ = (1 << num_bit_idd_);
    };

    bool ViewImpl::demux(bool only_push_keyframe){
        uint8_t* pdata;
        int size;
        uint64_t pts;
        bool iskey;
        bool ok = demuxer_->demux(&pdata, &size, &pts, &iskey);

        if(ok){
            if(only_push_keyframe && iskey || !only_push_keyframe){
                decoder_->push_demux(Packet(
                    idd_, pdata, size, iskey, pts
                ));
            }
        }
        return ok;
    }

    std::shared_ptr<Decoder> create_decoder(
        bool use_device_frame, int max_cache, int gpu_id, 
        const FFHDDecoder::CropRect *crop_rect, 
        const FFHDDecoder::ResizeDim *resize_dim, 
        FFHDDecoder::FrameType type
    ){
        return make_shared<DecoderImpl>(
            use_device_frame, max_cache, gpu_id, crop_rect, resize_dim, type
        );
    }

}// namespace FFHDMultiCamera;