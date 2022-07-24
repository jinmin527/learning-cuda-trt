
#ifndef KIWI_TENSOR_HPP
#define KIWI_TENSOR_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>

namespace kiwi {

#define CURRENT_DEVICE_ID           -1
    typedef struct{unsigned short _;} float16;

    enum class DataHead : int{
        Init   = 0,
        Device = 1,
        Host   = 2
    };

    enum class DataType : int {
        Float = 0,
        Float16 = 1,
        Int32 = 2,
        UInt8 = 3
    };

    int data_type_size(DataType dt);
    const char* data_head_string(DataHead dh);
    const char* data_type_string(DataType dt);

    class MixMemory {
    public:
        MixMemory(int device_id = CURRENT_DEVICE_ID);
        MixMemory(void* cpu, size_t cpu_size);
        virtual ~MixMemory();
        void* cpu(size_t size);
        void release_cpu();

        inline size_t cpu_size() const{return cpu_size_;}
        inline int device_id() const{return device_id_;}
        inline void* cpu() const { return cpu_; }

        void reference_data(void* cpu, size_t cpu_size);

    private:
        void* cpu_ = nullptr;
        size_t cpu_size_ = 0;
        bool owner_cpu_ = true;
        int device_id_ = 0;
    };

    class Tensor {
    public:
        Tensor(const Tensor& other) = delete;
        Tensor& operator = (const Tensor& other) = delete;

        explicit Tensor(DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int n, int c, int h, int w, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int ndims, const int* dims, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        virtual ~Tensor();

        int numel() const;
        inline int ndims() const{return shape_.size();}
        inline int size(int index)  const{return shape_[index];}
        inline int shape(int index) const{return shape_[index];}

        inline DataType type()                const { return dtype_; }
        inline const std::vector<int>& dims() const { return shape_; }
        inline const std::vector<size_t>& strides() const {return strides_;}
        inline int bytes()                    const { return bytes_; }
        inline int bytes(int start_axis)      const { return count(start_axis) * element_size(); }
        inline int element_size()             const { return data_type_size(dtype_); }
        inline DataHead head()                const { return head_; }

        Tensor& release();
        Tensor& set_to(float value);
        bool empty() const;

        template<typename ... _Args>
        int offset(int index, _Args ... index_args) const{
            const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
        }

        int offset_array(const std::vector<int>& index) const;
        int offset_array(size_t size, const int* index_array) const;

        template<typename ... _Args>
        Tensor& resize(int dim_size, _Args ... dim_size_args){
            const int dim_size_array[] = {dim_size, dim_size_args...};
            return resize(sizeof...(dim_size_args) + 1, dim_size_array);
        }

        Tensor& resize(int ndims, const int* dims);
        Tensor& resize(const std::vector<int>& dims);
        Tensor& resize_single_dim(int idim, int size);
        int  count(int start_axis = 0) const;
        int device() const{return device_id_;}
        inline void* cpu() const { return data_->cpu(bytes_); }

        template<typename DType> inline const DType* cpu() const { return (DType*)cpu(); }
        template<typename DType> inline DType* cpu()             { return (DType*)cpu(); }

        template<typename DType, typename ... _Args>
        inline DType* cpu(int i, _Args&& ... args) { return cpu<DType>() + offset(i, args...); }

        template<typename DType, typename ... _Args>
        inline DType& at(int i, _Args&& ... args) { return *(cpu<DType>() + offset(i, args...)); }

        std::shared_ptr<MixMemory> get_data()             const {return data_;}
        std::shared_ptr<MixMemory> get_workspace()        const {return workspace_;}
        Tensor& set_workspace(std::shared_ptr<MixMemory> workspace) {workspace_ = workspace; return *this;}

        const char* shape_string() const{return shape_string_;}
        const char* descriptor() const;

        void reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, DataType dtype);

        /**

        # 以下代码是python中加载Tensor
        import numpy as np

        def load_tensor(file):

            with open(file, "rb") as f:
                binary_data = f.read()

            magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
            assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

            dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

            if dtype == 0:
                np_dtype = np.float32
            elif dtype == 1:
                np_dtype = np.float16
            else:
                assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"

            return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)

         **/
        bool save_to_file(const std::string& file) const;
        bool load_from_file(const std::string& file);

    private:
        Tensor& compute_shape_string();
        Tensor& adajust_memory_by_update_dims_or_type();
        void setup_data(std::shared_ptr<MixMemory> data);

    private:
        std::vector<int> shape_;
        std::vector<size_t> strides_;
        size_t bytes_    = 0;
        DataHead head_   = DataHead::Init;
        DataType dtype_  = DataType::Float;
        int device_id_   = 0;
        char shape_string_[100];
        char descriptor_string_[100];
        std::shared_ptr<MixMemory> data_;
        std::shared_ptr<MixMemory> workspace_;
    };
};

#endif // KIWI_TENSOR_HPP