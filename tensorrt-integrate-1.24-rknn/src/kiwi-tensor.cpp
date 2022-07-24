
#include "kiwi-tensor.hpp"
#include "kiwi-logger.hpp"
#include <string.h>
#include <algorithm>
#include <assert.h>

using namespace std;

namespace kiwi{

    int data_type_size(DataType dt){
        switch (dt) {
            case DataType::Float: return sizeof(float);
            case DataType::Float16: return sizeof(float16);
            case DataType::Int32: return sizeof(int);
            case DataType::UInt8: return sizeof(uint8_t);
            default: {
                INFO("Not support dtype: %d", dt);
                return -1;
            }
        }
    }

    MixMemory::MixMemory(int device_id){
        device_id_ = device_id;
    }

    MixMemory::MixMemory(void* cpu, size_t cpu_size){
        reference_data(cpu, cpu_size);
    }

    void MixMemory::reference_data(void* cpu, size_t cpu_size){
        release_cpu();

        if(cpu == nullptr || cpu_size == 0){
            cpu = nullptr;
            cpu_size = 0;
        }

        this->cpu_ = cpu;
        this->cpu_size_ = cpu_size;
        this->owner_cpu_ = !(cpu && cpu_size > 0);
    }

    MixMemory::~MixMemory() {
        release_cpu();
    }

    void* MixMemory::cpu(size_t size) {

        if (cpu_size_ < size) {
            release_cpu();

            cpu_size_ = size;
            cpu_ = malloc(size);
            memset(cpu_, 0, size);
        }
        return cpu_;
    }

    void MixMemory::release_cpu() {
        if (cpu_) {
            if(owner_cpu_){
                free(cpu_);
            }
            cpu_ = nullptr;
        }
        cpu_size_ = 0;
    }

    const char* data_head_string(DataHead dh){
        switch(dh){
            case DataHead::Init: return "Init";
            case DataHead::Device: return "Device";
            case DataHead::Host: return "Host";
            default: return "Unknow";
        }
    }

    const char* data_type_string(DataType dt){
        switch(dt){
            case DataType::Float: return "Float32";
            case DataType::Float16: return "Float16";
            case DataType::Int32: return "Int32";
            case DataType::UInt8: return "UInt8";
            default: return "Unknow";
        }
    }

    Tensor::Tensor(int n, int c, int h, int w, DataType dtype, shared_ptr<MixMemory> data, int device_id) {
        this->dtype_ = dtype;
        this->device_id_ = device_id;
        descriptor_string_[0] = 0;
        setup_data(data);
        resize(n, c, h, w);
    }

    Tensor::Tensor(const std::vector<int>& dims, DataType dtype, shared_ptr<MixMemory> data, int device_id){
        this->dtype_ = dtype;
        this->device_id_ = device_id;
        descriptor_string_[0] = 0;
        setup_data(data);
        resize(dims);
    }

    Tensor::Tensor(int ndims, const int* dims, DataType dtype, shared_ptr<MixMemory> data, int device_id) {
        this->dtype_ = dtype;
        this->device_id_ = device_id;
        descriptor_string_[0] = 0;
        setup_data(data);
        resize(ndims, dims);
    }

    Tensor::Tensor(DataType dtype, shared_ptr<MixMemory> data, int device_id){
        shape_string_[0] = 0;
        descriptor_string_[0] = 0;
        this->device_id_ = device_id;
        dtype_ = dtype;
        setup_data(data);
    }

    Tensor::~Tensor() {
        release();
    }

    const char* Tensor::descriptor() const{

        char* descriptor_ptr = (char*)descriptor_string_;
        int device_id = device();
        snprintf(descriptor_ptr, sizeof(descriptor_string_),
                 "Tensor:%p, %s, %s, CUDA:%d",
                 data_.get(),
                 data_type_string(dtype_),
                 shape_string_,
                 device_id
        );
        return descriptor_ptr;
    }

    Tensor& Tensor::compute_shape_string(){

        // clean string
        shape_string_[0] = 0;

        char* buffer = shape_string_;
        size_t buffer_size = sizeof(shape_string_);
        for(int i = 0; i < shape_.size(); ++i){

            int size = 0;
            if(i < shape_.size() - 1)
                size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
            else
                size = snprintf(buffer, buffer_size, "%d", shape_[i]);

            buffer += size;
            buffer_size -= size;
        }
        return *this;
    }

    void Tensor::reference_data(const vector<int>& shape, void* cpu_data, size_t cpu_size, DataType dtype){

        dtype_ = dtype;
        data_->reference_data(cpu_data, cpu_size);
        setup_data(data_);
        resize(shape);
    }

    void Tensor::setup_data(shared_ptr<MixMemory> data){

        data_ = data;
        if(data_ == nullptr){
            data_ = make_shared<MixMemory>(device_id_);
        }else{
            device_id_ = data_->device_id();
        }

        head_ = DataHead::Init;
        if(data_->cpu()){
            head_ = DataHead::Host;
        }
    }

    Tensor& Tensor::release() {
        data_->release_cpu();
        shape_.clear();
        bytes_ = 0;
        head_ = DataHead::Init;
        return *this;
    }

    bool Tensor::empty() const{
        return data_->cpu() == nullptr;
    }

    int Tensor::count(int start_axis) const {

        if(start_axis >= 0 && start_axis < shape_.size()){
            int size = 1;
            for (int i = start_axis; i < shape_.size(); ++i)
                size *= shape_[i];
            return size;
        }else{
            return 0;
        }
    }

    Tensor& Tensor::resize(const std::vector<int>& dims) {
        return resize(dims.size(), dims.data());
    }

    int Tensor::numel() const{
        int value = shape_.empty() ? 0 : 1;
        for(int i = 0; i < shape_.size(); ++i){
            value *= shape_[i];
        }
        return value;
    }

    Tensor& Tensor::resize_single_dim(int idim, int size){

        auto new_shape = shape_;
        new_shape[idim] = size;
        return resize(new_shape);
    }

    Tensor& Tensor::resize(int ndims, const int* dims) {

        vector<int> setup_dims(ndims);
        for(int i = 0; i < ndims; ++i){
            int dim = dims[i];
            if(dim == -1){
                assert(ndims == shape_.size());
                dim = shape_[i];
            }
            setup_dims[i] = dim;
        }
        this->shape_ = setup_dims;

        // strides = element_size
        this->strides_.resize(setup_dims.size());

        size_t prev_size  = element_size();
        size_t prev_shape = 1;
        for(int i = (int)strides_.size() - 1; i >= 0; --i){
            if(i + 1 < strides_.size()){
                prev_size  = strides_[i+1];
                prev_shape = shape_[i+1];
            }
            strides_[i] = prev_size * prev_shape;
        }

        this->adajust_memory_by_update_dims_or_type();
        this->compute_shape_string();
        return *this;
    }

    Tensor& Tensor::adajust_memory_by_update_dims_or_type(){

        int needed_size = this->numel() * element_size();
        if(needed_size > this->bytes_){
            head_ = DataHead::Init;
        }
        this->bytes_ = needed_size;
        return *this;
    }

    template<typename _T>
    static inline void memset_any_type(_T* ptr, size_t count, _T value){
        for (size_t i = 0; i < count; ++i)
            *ptr++ = value;
    }

    Tensor& Tensor::set_to(float value) {
        int c = count();
        if (dtype_ == DataType::Float) {
            memset_any_type(cpu<float>(), c, value);
        }
        else if(dtype_ == DataType::Int32) {
            memset_any_type(cpu<int>(), c, (int)value);
        }
        else if(dtype_ == DataType::UInt8) {
            memset_any_type(cpu<uint8_t>(), c, (uint8_t)value);
        }
        else{
            INFO("Unsupport type: %d", dtype_);
        }
        return *this;
    }

    int Tensor::offset_array(size_t size, const int* index_array) const{

        assert(size <= shape_.size());
        int value = 0;
        for(int i = 0; i < shape_.size(); ++i){

            if(i < size)
                value += index_array[i];

            if(i + 1 < shape_.size())
                value *= shape_[i+1];
        }
        return value;
    }

    int Tensor::offset_array(const std::vector<int>& index_array) const{
        return offset_array(index_array.size(), index_array.data());
    }

    bool Tensor::save_to_file(const std::string& file) const{

        if(empty()) return false;

        FILE* f = fopen(file.c_str(), "wb");
        if(f == nullptr) return false;

        unsigned int ndims = this->ndims();
        unsigned int head[3] = {0xFCCFE2E2, ndims, static_cast<unsigned int>(dtype_)};
        fwrite(head, 1, sizeof(head), f);
        fwrite(shape_.data(), 1, sizeof(shape_[0]) * shape_.size(), f);
        fwrite(cpu(), 1, bytes_, f);
        fclose(f);
        return true;
    }

    bool Tensor::load_from_file(const std::string& file){

        FILE* f = fopen(file.c_str(), "rb");
        if(f == nullptr){
            INFOE("Open %s failed.", file.c_str());
            return false;
        }

        unsigned int head[3] = {0};
        fread(head, 1, sizeof(head), f);

        if(head[0] != 0xFCCFE2E2){
            fclose(f);
            INFOE("Invalid tensor file %s, magic number mismatch", file.c_str());
            return false;
        }

        int ndims = head[1];
        auto dtype = (kiwi::DataType)head[2];
        vector<int> dims(ndims);
        fread(dims.data(), 1, ndims * sizeof(dims[0]), f);

        this->dtype_ = dtype;
        this->resize(dims);

        fread(this->cpu(), 1, bytes_, f);
        fclose(f);
        return true;
    }

}; // TRTTensor