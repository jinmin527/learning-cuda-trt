#!/bin/bash

pip install -r requirements.txt

cd insightface-master/python-package/insightface/thirdparty/face3d/mesh/cython
python setup.py build

sopath=`find build -name *.so`

if [ ! -f ${sopath} ]; then
    echo Build module failed. sopath=[${sopath}]
    exit
fi

mv ${sopath} ./
rm -rf build