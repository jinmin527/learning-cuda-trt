
#include "mix-memory.hpp"
#include "cuda-tools.hpp"
#include <string.h>
#include <assert.h>

namespace TRT{

    inline static int check_and_trans_device_id(int device_id){
        if(device_id != CURRENT_DEVICE_ID){
            CUDATools::check_device_id(device_id);
            return device_id;
        }

        checkRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    MixMemory::MixMemory(int device_id){
        device_id_ = check_and_trans_device_id(device_id);
    }

    MixMemory::MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size, int device_id){
        reference_data(cpu, cpu_size, gpu, gpu_size, device_id);		
    }

    void MixMemory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size, int device_id){
        release_all();
        
        if(cpu == nullptr || cpu_size == 0){
            cpu = nullptr;
            cpu_size = 0;
        }

        if(gpu == nullptr || gpu_size == 0){
            gpu = nullptr;
            gpu_size = 0;
        }

        this->cpu_ = cpu;
        this->cpu_size_ = cpu_size;
        this->gpu_ = gpu;
        this->gpu_size_ = gpu_size;

        this->owner_cpu_ = !(cpu && cpu_size > 0);
        this->owner_gpu_ = !(gpu && gpu_size > 0);
        device_id_ = check_and_trans_device_id(device_id);
    }

    MixMemory::~MixMemory() {
        release_all();
    }

    void* MixMemory::gpu(size_t size) {

        if (gpu_size_ < size) {
            release_gpu();

            gpu_size_ = size;
            CUDATools::AutoDevice auto_device_exchange(device_id_);
            checkRuntime(cudaMalloc(&gpu_, size));
            checkRuntime(cudaMemset(gpu_, 0, size));
        }
        return gpu_;
    }

    void* MixMemory::cpu(size_t size) {

        if (cpu_size_ < size) {
            release_cpu();

            cpu_size_ = size;
            CUDATools::AutoDevice auto_device_exchange(device_id_);
            checkRuntime(cudaMallocHost(&cpu_, size));
            assert(cpu_ != nullptr);
            memset(cpu_, 0, size);
        }
        return cpu_;
    }

    void MixMemory::release_cpu() {
        if (cpu_) {
            if(owner_cpu_){
                CUDATools::AutoDevice auto_device_exchange(device_id_);
                checkRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_size_ = 0;
    }

    void MixMemory::release_gpu() {
        if (gpu_) {
            if(owner_gpu_){
                CUDATools::AutoDevice auto_device_exchange(device_id_);
                checkRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_size_ = 0;
    }

    void MixMemory::release_all() {
        release_cpu();
        release_gpu();
    }
};