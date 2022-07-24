#!/bin/bash

cd insightface-master
files=`find . -name __pycache__`

for file in ${files};
do
    echo Remove ${file}
    rm -rf ${file}
done

echo Remove python-package/insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.*.so
rm -rf python-package/insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.*.so