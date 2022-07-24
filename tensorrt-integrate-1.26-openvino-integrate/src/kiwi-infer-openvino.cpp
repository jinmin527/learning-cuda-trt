

#include "kiwi-infer-openvino.hpp"
#include "kiwi-logger.hpp"
#include <algorithm>
#include <fstream>

#ifdef WITH_OPENVINO
#include "openvino/openvino.hpp"

namespace openvino {

    using namespace std;
    using namespace kiwi;

    ////////////////////////////////////////////////////////////////////////////////
    class OVModel{
    public:
        ov::Core handler_;
        ov::CompiledModel model_;
        ov::InferRequest iq_;
        std::vector<ov::Tensor> ov_inputs_;
        std::vector<ov::Tensor> ov_outputs_;

        bool load_model(const string& file){
            try{
                model_ = handler_.compile_model(file);
                iq_ = model_.create_infer_request();

                auto inputs = model_.inputs();
                auto outputs = model_.outputs();
                ov_inputs_.resize(inputs.size());
                ov_outputs_.resize(outputs.size());
                for(int i = 0; i < inputs.size(); ++i){
                    ov_inputs_[i] = iq_.get_tensor(inputs[i]);
                    if(!ov_inputs_[i].get_element_type().is_static()){
                        INFOE("Only support static shape model");
                        return false;
                    }
                }
                
                for(int i = 0; i < outputs.size(); ++i){
                    ov_outputs_[i] = iq_.get_tensor(outputs[i]);
                    if(!ov_outputs_[i].get_element_type().is_static()){
                        INFOE("Only support static shape model");
                        return false;
                    }
                }

                return true;
            }catch(ov::Exception ex){
                INFOE("Load openvino model %s failed: %s", file.c_str(), ex.what());
            }
            return false;
        }
    };

    class InferImpl : public Infer {

    public:
        virtual ~InferImpl();
        virtual bool load(const std::string& file);
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
        std::shared_ptr<OVModel> ovmodel_;
    };

    ////////////////////////////////////////////////////////////////////////////////////
    InferImpl::~InferImpl(){
        destroy();
    }

    void InferImpl::destroy() {

        this->blobsNameMapper_.clear();
        this->outputs_.clear();
        this->inputs_.clear();
        this->inputs_name_.clear();
        this->outputs_name_.clear();
        this->ovmodel_.reset();
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

    bool InferImpl::load(const std::string& file) {

        this->ovmodel_.reset(new OVModel());
        if(!this->ovmodel_->load_model(file))
            return false;

        workspace_.reset(new MixMemory());
        return build_engine_input_and_outputs_mapper();
    }

    static kiwi::DataType convert_trt_datatype(ov::element::Type dt){
        switch(dt){
            case ov::element::Type_t::f32: return kiwi::DataType::Float;
            case ov::element::Type_t::f16: return kiwi::DataType::Float16;
            case ov::element::Type_t::u8: return kiwi::DataType::UInt8;
            default:
                INFOE("Unsupport data type %s", dt.get_type_name().c_str());
                return kiwi::DataType::Float;
        }
    }

    template<typename _T>
    vector<int> convert_shape(const vector<_T>& shape){
        vector<int> new_shape;
        for(auto i : shape)
            new_shape.push_back(i);
        return new_shape;
    }

    bool InferImpl::build_engine_input_and_outputs_mapper() {

        inputs_.clear();
        inputs_name_.clear();
        outputs_.clear();
        outputs_name_.clear();
        orderdBlobs_.clear();
        blobsNameMapper_.clear();

        for(int i = 0; i < ovmodel_->ov_inputs_.size(); ++i){

            auto run_input = ovmodel_->ov_inputs_[i];
            auto desc_input = ovmodel_->model_.input(i);
            auto bindingName = desc_input.get_any_name();
            auto shape = convert_shape(run_input.get_shape());
            auto mem = make_shared<MixMemory>(run_input.data(), run_input.get_byte_size());
            auto dtype = convert_trt_datatype(run_input.get_element_type());
            auto newTensor = make_shared<Tensor>(shape, dtype, mem);

            newTensor->set_workspace(this->workspace_);
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
            inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);
        }

        for(int i = 0; i < ovmodel_->ov_outputs_.size(); ++i){

            auto run_output = ovmodel_->ov_outputs_[i];
            auto desc_output = ovmodel_->model_.output(i);
            auto bindingName = desc_output.get_any_name();
            auto shape = convert_shape(run_output.get_shape());
            auto mem = make_shared<MixMemory>(run_output.data(), run_output.get_byte_size());
            auto dtype = convert_trt_datatype(run_output.get_element_type());
            auto newTensor = make_shared<Tensor>(shape, dtype, mem);

            newTensor->set_workspace(this->workspace_);
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
            outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
            blobsNameMapper_[bindingName] = i;
            orderdBlobs_.push_back(newTensor);
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
        ovmodel_->iq_.infer();
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

    std::shared_ptr<Infer> load_infer(const string& file) {

        std::shared_ptr<InferImpl> infer(new InferImpl());
        if (!infer->load(file))
            infer.reset();
        return infer;
    }
};  // openvino
#else

namespace openvino{
    
    std::shared_ptr<kiwi::Infer> load_infer(const std::string& file){
        INFOE("Unimplement backend OpenVINO.");
        return nullptr;
    }

}; // openvino
#endif // WITH_OPENVINO