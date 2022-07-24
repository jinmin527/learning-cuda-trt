#ifndef KIWI_INFER_HPP
#define KIWI_INFER_HPP

#include "kiwi-tensor.hpp"

namespace kiwi{

    enum class Backend : int{
        None = 0,
        RKNN = 1,
        OpenVINO = 2
    };

    class Infer {
    public:
        virtual bool     forward() = 0;
        virtual std::shared_ptr<MixMemory> get_workspace() = 0;
        virtual std::shared_ptr<Tensor>    input (int index = 0) = 0;
        virtual std::shared_ptr<Tensor>    output(int index = 0) = 0;
        virtual std::shared_ptr<Tensor>    tensor(const std::string& name) = 0;
        virtual std::string get_input_name (int index = 0) = 0;
        virtual std::string get_output_name(int index = 0) = 0;
        virtual bool is_output_name(const std::string& name) = 0;
        virtual bool is_input_name (const std::string& name) = 0;
        virtual int  num_output() = 0;
        virtual int  num_input() = 0;
        virtual void print() = 0;
        virtual void set_input (int index, std::shared_ptr<Tensor> tensor) = 0;
        virtual void set_output(int index, std::shared_ptr<Tensor> tensor) = 0;
    };  

    Backend set_backend(Backend backend);
    Backend get_backend();
    
    std::shared_ptr<kiwi::Infer> load_infer_with_backend(Backend backend, const std::string& file);
    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file);
}; // namespace kiwi

#endif // KIWI_INFER_HPP