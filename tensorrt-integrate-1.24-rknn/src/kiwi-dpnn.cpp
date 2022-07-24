

#include "kiwi-dpnn.hpp"
#include "kiwi-logger.hpp"
#include <fstream>
#include <memory>
#include <string.h>

namespace kiwi{

    using namespace std;

    bool DPNN::has_label(int label){
        return this->labels.find(label) != this->labels.end();
    }

    std::string DPNN::label_to_name(int label){
        auto iter = this->labels.find(label);
        if(iter == this->labels.end()){
            return "";
        }
        return iter->second;
    }

    const char* model_type_string(ModelType type){
        switch(type){
            case ModelType::RKNN: return "RKNN";
            case ModelType::DPNN: return "DPNN";
            case ModelType::Unknow: return "Unknow";
            default: return "Unknow";
        }
    }

    ModelType get_model_format(const std::string& file){

        FILE* f = fopen(file.c_str(), "rb");
        if(f == nullptr){
            INFOE("Load model failed, %s", file.c_str());
            return ModelType::Unknow;
        }

        int head = 0;
        if(fread(&head, 1, sizeof(head), f) != sizeof(head)){
            INFOE("Invalid model file.");
            fclose(f);
            return ModelType::Unknow;
        }
        fclose(f);

        if(head == 0x33ffcc15){
            return ModelType::DPNN;
        }else if(head == 0x4E4E4B52){  // RKNN
            return ModelType::RKNN;
        }
        return ModelType::Unknow;
    }

    bool load_dpnn(const std::string& file, DPNN& dpnn){
        
        FILE* f = fopen(file.c_str(), "rb");
        if(f == nullptr){
            INFOE("Load model failed, %s", file.c_str());
            return false;
        }

        shared_ptr<FILE> f_shared(f, fclose);
        unsigned int head[3];
        if(fread(head, 1, sizeof(head), f) != sizeof(head)){
            INFOE("invalid dpnn model.");
            return false;
        }

        unsigned int magic_number = head[0];
        unsigned int meta_length = head[1];
        unsigned int model_length = head[2];
        if(magic_number != 0x33ffcc15){
            INFOE("Invalid magic number %X", magic_number);
            return false;
        }

        vector<char> meta(meta_length);
        if(fread(meta.data(), 1, meta.size(), f) != meta_length){
            INFOE("Meta data is too short.");
            return false;
        }
 
        dpnn.data.resize(model_length);
        if(fread(&dpnn.data[0], 1, model_length, f) != model_length){
            INFOE("Model data is too short.");
            return false;
        }

        auto metaj = Json::parse_string(meta.data());
        if(!metaj.isMember("infer-type")){
            INFOE("No member infer-type in meta");
            return false;
        }

        try{
            dpnn.infer_type = metaj.get("infer-type", "").asString();
            dpnn.name = metaj.get("name", "").asString();
            dpnn.artist = metaj.get("artist", "").asString();
            dpnn.version = metaj.get("version", "").asString();
            dpnn.date_time = metaj.get("date-time", "").asString();
            dpnn.params = metaj.get("params", Json::objectValue);

            dpnn.labels.clear();
            if(metaj.isMember("labels")){
                auto& labels = metaj["labels"];
                if(labels.isObject()){
                    auto members = labels.getMemberNames();
                    for(int i = 0; i < members.size(); ++i){
                        auto& name = members[i];
                        dpnn.labels[atoi(name.c_str())] = labels.get(name, "").asString();
                    }
                }else{
                    INFOW("Labels not a dict");
                }
            }
            return true;
        }catch(Json::Exception e){
            INFOE("Parse meta failed: %s", e.what());
            return false;
        }
        return false;
    }
}; // kiwi