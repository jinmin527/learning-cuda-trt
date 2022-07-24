#ifndef KIWI_INFER_RKNN_HPP
#define KIWI_INFER_RKNN_HPP

#include "kiwi-infer.hpp"

namespace rknn{

    /*
        sync mode对应RKNN_FLAG_ASYNC_MASK标记，该标记设置后，forward后，拿到的结果是上一帧的结果（第一帧除外）
        作用是实现时间重叠，如果无法利用该特性，请关闭sync_mode
        该模式仅在需要立即大量推理的场景能够使得性能提升20%左右，充分利用NPU
            例如要一次处理1000个图的时候，可以利用该特性获得性能提升
        如果是摄像头模式下，需要跟摄像头时间重叠，则可以利用该特性
    */
    std::shared_ptr<kiwi::Infer> load_infer_from_memory(const void* pdata, size_t size, bool sync_mode=false);
    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file_or_data, bool sync_mode=false);

}; // namespace kiwi

#endif // KIWI_INFER_RKNN_HPP