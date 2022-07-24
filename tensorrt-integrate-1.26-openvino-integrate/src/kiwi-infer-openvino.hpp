#ifndef KIWI_INFER_OPENVINO_HPP
#define KIWI_INFER_OPENVINO_HPP

#include "kiwi-infer.hpp"

namespace openvino{

    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file);

}; // namespace openvino

#endif // KIWI_INFER_OPENVINO_HPP