#!/bin/bash

# enable trtpy library environment
# source environment.sh
export LD_LIBRARY_PATH=${@NVLIB64}:${@PYTHON_LIB}:${@TRTPRO_LIB}:${@CPP_PKG}/opencv4.2/lib:${@SYS_LIB}:$LD_LIBRARY_PATH
export PATH=${@CUDA_HOME}/bin:$PATH

if [ ! -d "${@CUDA_HOME}/phony" ]; then
    echo Make phony CUDA_HOME=${@CUDA_HOME}/phony
    mkdir -p "${@CUDA_HOME}/phony"

    echo Create soft link
    ln -s "${@CUDA_HOME}/bin" "${@CUDA_HOME}/phony/bin"
    ln -s "${@NVLIB64}" "${@CUDA_HOME}/phony/lib64"
    ln -s "${@CUDA_INCLUDE}" "${@CUDA_HOME}/phony/include"
    ln -s "${@CUDA_HOME}/nvvm" "${@CUDA_HOME}/phony/nvvm"
    ln -s "${@CUDA_HOME}/extras" "${@CUDA_HOME}/phony/extras"
fi

export CUDA_HOME=${@CUDA_HOME}/phony
