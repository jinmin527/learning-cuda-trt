#ifndef KIWI_DPNN_HPP
#define KIWI_DPNN_HPP

#include <kiwi-json.hpp>
#include <string>
#include <map>

// 是一种带有meta元信息的rknn模型封装格式
// 以下是生成模型的方式，通过带有meta信息，我们可以区分模型是否正确
// 用来检验是否使用了正确的模型和推理器
/*
import numpy as np
import json

meta = {
    "name": "w600k-r50",
    "infer-type": "arcface",
    "artist": "insightface",
    "version": "1",
    "date-time": "2022-3-15 13:53:39",
    "labels": {},
    "params": {}
}

def to_dpnn(model_name, meta, output):

    with open(output, "wb") as m:
        meta_bytes = json.dumps(meta).encode("utf-8")
        rknn_data = open(model_name, "rb").read()
        head = [0x33ffcc15, len(meta_bytes), len(rknn_data)]
        m.write(np.array(head, dtype=np.int32).tobytes())
        m.write(meta_bytes)
        m.write(rknn_data)

to_dpnn(
    model_name = "workspace/w600k_r50_new.rknn",
    meta = meta,
    output = "workspace/w600k_r50_new.dpnn"
)
*/
namespace kiwi{

    enum class ModelType : int{
        Unknow = 0,
        DPNN = 1,
        RKNN = 2
    };

    struct DPNN{
        std::string name;
        std::string infer_type;
        std::string artist;
        std::string version;
        std::string date_time;
        std::map<int, std::string> labels;
        Json::Value params;
        std::string data;

        std::string label_to_name(int label);
        bool has_label(int label);
    };

    const char* model_type_string(ModelType type);
    bool load_dpnn(const std::string& file, DPNN& dpnn);
    ModelType get_model_format(const std::string& file);
};

#endif // KIWI_DPNN_HPP