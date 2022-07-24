
#include "kiwi-infer-rknn.hpp"
#include "rknn_api.h"
#include "kiwi-logger.hpp"
#include <algorithm>
#include <fstream>

namespace rknn {

    using namespace std;
    using namespace kiwi;

    #define checkRKNN(op)   rknn::__checkRKNN((op), #op, __FILE__, __LINE__)

    #define RKNN_SUCC                               0       /* execute succeed. */
    #define RKNN_ERR_FAIL                           -1      /* execute failed. */
    #define RKNN_ERR_TIMEOUT                        -2      /* execute timeout. */
    #define RKNN_ERR_DEVICE_UNAVAILABLE             -3      /* device is unavailable. */
    #define RKNN_ERR_MALLOC_FAIL                    -4      /* memory malloc fail. */
    #define RKNN_ERR_PARAM_INVALID                  -5      /* parameter is invalid. */
    #define RKNN_ERR_MODEL_INVALID                  -6      /* model is invalid. */
    #define RKNN_ERR_CTX_INVALID                    -7      /* context is invalid. */
    #define RKNN_ERR_INPUT_INVALID                  -8      /* input is invalid. */
    #define RKNN_ERR_OUTPUT_INVALID                 -9      /* output is invalid. */
    #define RKNN_ERR_DEVICE_UNMATCH                 -10     /* the device is unmatch, please update rknn sdk and npu driver/firmware. */
    #define RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL  -11     /* This RKNN model use pre_compile mode, but not compatible with current driver. */
    #define RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION  -12     /* This RKNN model set optimization level, but not compatible with current driver. */
    #define RKNN_ERR_TARGET_PLATFORM_UNMATCH        -13     /* This RKNN model set target platform, but not compatible with current platform. */

    static const char* get_code_name(int code){
        switch(code){
            case RKNN_SUCC: return "RKNN_SUCC";
            case RKNN_ERR_FAIL: return "RKNN_ERR_FAIL";
            case RKNN_ERR_TIMEOUT: return "RKNN_ERR_TIMEOUT";
            case RKNN_ERR_DEVICE_UNAVAILABLE: return "RKNN_ERR_DEVICE_UNAVAILABLE";
            case RKNN_ERR_MALLOC_FAIL: return "RKNN_ERR_MALLOC_FAIL";
            case RKNN_ERR_PARAM_INVALID: return "RKNN_ERR_PARAM_INVALID";
            case RKNN_ERR_MODEL_INVALID: return "RKNN_ERR_MODEL_INVALID";
            case RKNN_ERR_CTX_INVALID: return "RKNN_ERR_CTX_INVALID";
            case RKNN_ERR_INPUT_INVALID: return "RKNN_ERR_INPUT_INVALID";
            case RKNN_ERR_OUTPUT_INVALID: return "RKNN_ERR_OUTPUT_INVALID";
            case RKNN_ERR_DEVICE_UNMATCH: return "RKNN_ERR_DEVICE_UNMATCH";
            case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
            case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION: return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
            case RKNN_ERR_TARGET_PLATFORM_UNMATCH: return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
            default:
                INFOW("Unknow Code: %d", code);
                return "UnknowCode";
        }
    }

    static const char* get_code_descript(int code){
        switch(code){
            case RKNN_SUCC: return "execute succeed.";
            case RKNN_ERR_FAIL: return "execute failed.";
            case RKNN_ERR_TIMEOUT: return "execute timeout.";
            case RKNN_ERR_DEVICE_UNAVAILABLE: return "device is unavailable.";
            case RKNN_ERR_MALLOC_FAIL: return "memory malloc fail.";
            case RKNN_ERR_PARAM_INVALID: return "parameter is invalid.";
            case RKNN_ERR_MODEL_INVALID: return "model is invalid.";
            case RKNN_ERR_CTX_INVALID: return "context is invalid.";
            case RKNN_ERR_INPUT_INVALID: return "input is invalid.";
            case RKNN_ERR_OUTPUT_INVALID: return "output is invalid.";
            case RKNN_ERR_DEVICE_UNMATCH: return "the device is unmatch, please update rknn sdk and npu driver/firmware.";
            case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: return "This RKNN model use pre_compile mode, but not compatible with current driver.";
            case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION: return "This RKNN model set optimization level, but not compatible with current driver.";
            case RKNN_ERR_TARGET_PLATFORM_UNMATCH: return "This RKNN model set target platform, but not compatible with current platform.";
            default:
                INFOW("Unknow Code: %d", code);
                return "Unknow Code";
        }
    }

    static bool __checkRKNN(int code, const char* op, const char* file, int line){

        if(code < 0){
            const char* err_name = get_code_name(code);
            const char* err_message = get_code_descript(code);
            INFOE("rknn error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
            return false;
        }
        return true;
    }

    
    ////////////////////////////////////////////////////////////////////////////////
    class InferImpl : public Infer {

    public:
        virtual ~InferImpl();
        virtual bool load(const std::string& file, bool sync_mode);
        virtual bool load_from_memory(const void* pdata, size_t size, bool sync_mode);
        virtual void destroy();
        virtual bool forward() override;
        virtual std::shared_ptr<MixMemory> get_workspace() override;
        virtual std::shared_ptr<Tensor> input(int index = 0) override;
        virtual std::string get_input_name(int index = 0) override;
        virtual std::shared_ptr<Tensor> output(int index = 0) override;
        virtual std::string get_output_name(int index = 0) override;
        virtual std::shared_ptr<Tensor> tensor(const std::string& name) override;
        virtual bool is_output_name(const std::string& name) override;
        virtual bool is_input_name(const std::string& name) override;
        virtual void set_input (int index, std::shared_ptr<Tensor> tensor) override;
        virtual void set_output(int index, std::shared_ptr<Tensor> tensor) override;

        virtual void print() override;

        virtual int num_output() override;
        virtual int num_input() override;

    private:
        bool build_engine_input_and_outputs_mapper();

    private:
        std::vector<std::shared_ptr<Tensor>> inputs_;
        std::vector<std::shared_ptr<Tensor>> outputs_;
        std::vector<int> inputs_map_to_ordered_index_;
        std::vector<int> outputs_map_to_ordered_index_;
        std::vector<std::string> inputs_name_;
        std::vector<std::string> outputs_name_;
        std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
        std::map<std::string, int> blobsNameMapper_;
        std::shared_ptr<MixMemory> workspace_;
        std::vector<rknn_input> rknn_input_struct_;
        std::vector<rknn_output> rknn_output_struct_;
        rknn_context rknn_handle_ = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////
    InferImpl::~InferImpl(){
        destroy();
    }

    void InferImpl::destroy() {

        int old_device = 0;
        this->blobsNameMapper_.clear();
        this->outputs_.clear();
        this->inputs_.clear();
        this->inputs_name_.clear();
        this->outputs_name_.clear();
        this->rknn_input_struct_.clear();
        this->rknn_output_struct_.clear();

        if(rknn_handle_) {
            checkRKNN(::rknn_destroy(rknn_handle_));
            rknn_handle_ = 0;
        }
    }

    void InferImpl::print(){
        INFO("Infer %p detail", this);
        INFO("\tInputs: %d", inputs_.size());
        for(int i = 0; i < inputs_.size(); ++i){
            auto& tensor = inputs_[i];
            auto& name = inputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }

        INFO("\tOutputs: %d", outputs_.size());
        for(int i = 0; i < outputs_.size(); ++i){
            auto& tensor = outputs_[i];
            auto& name = outputs_name_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }
    }

    bool InferImpl::load_from_memory(const void* pdata, size_t size, bool sync_mode) {

        if (pdata == nullptr || size == 0)
            return false;

        INFO("RKNN init");
        int flags = RKNN_FLAG_PRIOR_MEDIUM;
        if(sync_mode){
            flags |= RKNN_FLAG_ASYNC_MASK;
        }
        if(!checkRKNN(::rknn_init(&rknn_handle_, (void*)pdata, size, flags))){
            INFOE("Load model failed, size = %d", size);
            return false;
        }
        INFO("RKNN init done.!");

        workspace_.reset(new MixMemory());
        return build_engine_input_and_outputs_mapper();
    }

    bool InferImpl::load(const std::string& file, bool sync_mode) {

        auto data = load_file(file);
        if (data.empty()){
            INFOE("Load file empty: %s", file.c_str());
            return false;
        }
        return load_from_memory(data.data(), data.size(), sync_mode);
    }

    static kiwi::DataType convert_trt_datatype(rknn_tensor_type dt){
        switch(dt){
            case rknn_tensor_type::RKNN_TENSOR_FLOAT32: return kiwi::DataType::Float;
            case rknn_tensor_type::RKNN_TENSOR_FLOAT16: return kiwi::DataType::Float16;
            case rknn_tensor_type::RKNN_TENSOR_UINT8: return kiwi::DataType::UInt8;
            default:
                INFOE("Unsupport data type %d", dt);
                return kiwi::DataType::Float;
        }
    }

    bool InferImpl::build_engine_input_and_outputs_mapper() {

        rknn_input_output_num inout_num;
        if(!checkRKNN(rknn_query(rknn_handle_, RKNN_QUERY_IN_OUT_NUM, &inout_num, sizeof(inout_num)))){
            INFOE("Query number io failed");
            return false;
        }

        inputs_.clear();
        inputs_name_.clear();
        outputs_.clear();
        outputs_name_.clear();
        orderdBlobs_.clear();
        blobsNameMapper_.clear();

        rknn_input_struct_.resize(inout_num.n_input);
        rknn_output_struct_.resize(inout_num.n_output);

        for(int i = 0; i < inout_num.n_input; ++i){
            rknn_tensor_attr input_attr;
            input_attr.index = i;
            if(!checkRKNN(rknn_query(rknn_handle_, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)))){
                INFOE("Query input %d failed", i);
                return false;
            }
            
            const char* bindingName = input_attr.name;
            std::vector<int> shape((int*)input_attr.dims, (int*)input_attr.dims + input_attr.n_dims);
            std::reverse(shape.begin(), shape.end());
            auto newTensor = make_shared<Tensor>(shape, convert_trt_datatype(input_attr.type));
            //auto newTensor = make_shared<Tensor>(input_attr.n_dims, (int*)input_attr.dims, convert_trt_datatype(input_attr.type));
            newTensor->set_workspace(this->workspace_);
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
            inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);

            rknn_input& input = rknn_input_struct_[i];
            input.index = i;
            input.buf = newTensor->cpu();
            input.size = newTensor->bytes();
            input.pass_through = false;
            input.type = RKNN_TENSOR_UINT8;
            input.fmt = RKNN_TENSOR_NHWC;
        }

        for(int i = 0; i < inout_num.n_output; ++i){
            rknn_tensor_attr output_attr;
            output_attr.index = i;
            if(!checkRKNN(rknn_query(rknn_handle_, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr)))){
                INFOE("Query output %d failed", i);
                return false;
            }
            output_attr.type = rknn_tensor_type::RKNN_TENSOR_FLOAT32;
            const char* bindingName = output_attr.name;
            std::vector<int> shape((int*)output_attr.dims, (int*)output_attr.dims + output_attr.n_dims);
            std::reverse(shape.begin(), shape.end());
            auto newTensor = make_shared<Tensor>(shape, convert_trt_datatype(output_attr.type));
            newTensor->set_workspace(this->workspace_);
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
            outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);

            rknn_output& output = rknn_output_struct_[i];
            output.index = i;
            output.want_float = true;
            output.is_prealloc = true;
            output.buf = newTensor->cpu();
            output.size = newTensor->bytes();
        }
        return true;
    }

    bool InferImpl::is_output_name(const std::string& name){
        return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
    }

    bool InferImpl::is_input_name(const std::string& name){
        return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
    }

    bool InferImpl::forward() {
        if(!checkRKNN(::rknn_inputs_set(rknn_handle_, rknn_input_struct_.size(), rknn_input_struct_.data()))){
            INFOE("Set input failed.");
            return false;
        }
        if(!checkRKNN(::rknn_run(rknn_handle_, nullptr))){
            INFOE("Run failed.");
            return false;
        }
        if(!checkRKNN(::rknn_outputs_get(rknn_handle_, rknn_output_struct_.size(), rknn_output_struct_.data(), nullptr))){
            INFOE("Get output failed.");
            return false;
        }
        // if(!checkRKNN(::rknn_outputs_release(rknn_handle_, rknn_output_struct_.size(), rknn_output_struct_.data()))){
        //     INFOE("Release output failed.");
        //     return false;
        // }
        return true;
    }

    std::shared_ptr<MixMemory> InferImpl::get_workspace() {
        return workspace_;
    }

    int InferImpl::num_input() {
        return static_cast<int>(this->inputs_.size());
    }

    int InferImpl::num_output() {
        return static_cast<int>(this->outputs_.size());
    }

    void InferImpl::set_input (int index, std::shared_ptr<Tensor> tensor){

        if(index < 0 || index >= inputs_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        }

        this->inputs_[index] = tensor;
        int order_index = inputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor){

        if(index < 0 || index >= outputs_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        }

        this->outputs_[index] = tensor;
        int order_index = outputs_map_to_ordered_index_[index];
        this->orderdBlobs_[order_index] = tensor;
    }

    std::shared_ptr<Tensor> InferImpl::input(int index) {
        if(index < 0 || index >= inputs_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        }
        return this->inputs_[index];
    }

    std::string InferImpl::get_input_name(int index){
        if(index < 0 || index >= inputs_name_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_name_.size());
        }
        return inputs_name_[index];
    }

    std::shared_ptr<Tensor> InferImpl::output(int index) {
        if(index < 0 || index >= outputs_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        }
        return outputs_[index];
    }

    std::string InferImpl::get_output_name(int index){
        if(index < 0 || index >= outputs_name_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_name_.size());
        }
        return outputs_name_[index];
    }

    std::shared_ptr<Tensor> InferImpl::tensor(const std::string& name) {

        auto node = this->blobsNameMapper_.find(name);
        if(node == this->blobsNameMapper_.end()){
            INFOF("Could not found the input/output node '%s', please makesure your model", name.c_str());
        }
        return orderdBlobs_[node->second];
    }

    std::shared_ptr<Infer> load_infer_from_memory(const void* pdata, size_t size, bool sync_mode){

        std::shared_ptr<InferImpl> infer(new InferImpl());
        if (!infer->load_from_memory(pdata, size, sync_mode))
            infer.reset();
        return infer;
    }

    static bool string_is_rknn_data(const string& s){

        bool find_zero = false;
        const char* p = s.c_str();
        for(int i = 0; i < s.size(); ++i, ++p){
            if(*p == 0) find_zero = true;
        }
        return find_zero;
    }

    std::shared_ptr<Infer> load_infer(const string& file_or_data, bool sync_mode) {

        std::shared_ptr<InferImpl> infer(new InferImpl());
        bool is_rknn_data = string_is_rknn_data(file_or_data);
        if(is_rknn_data){
            if (!infer->load_from_memory(file_or_data.data(), file_or_data.size(), sync_mode))
                infer.reset();
        }else{
            if (!infer->load(file_or_data, sync_mode))
                infer.reset();
        }
        return infer;
    }
}; 