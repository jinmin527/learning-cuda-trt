
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

vector<string> load_lines(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

unordered_map<string, int> load_vocab(const string& file, bool add_lower_case=true){

    unordered_map<string, int> vocab;
    auto lines = load_lines(file.c_str());
    for(int i = 0; i < lines.size(); ++i){
        auto token = lines[i];
        vocab[token] = i;

        if(add_lower_case){
            if(!token.empty() && token[0] != '[' && token.back() != ']'){
                for(int j = 0; j < token.size(); ++j){
                    if(token[j] >= 'A' && token[j] <= 'Z')
                        token[j] = token[j] - 'A' + 'a';
                }
            }
            if(vocab.find(token) == vocab.end())
                vocab[token] = i;
        }
    }
    return vocab;
}

int find_token(const string& token, const unordered_map<string, int>& vocab){
    auto iter = vocab.find(token);
    if(iter == vocab.end()){
        if(!token.empty() && token[0] != '[' && token.back() != ']'){
            string new_token = token;
            bool has_upper = false;
            for(int j = 0; j < new_token.size(); ++j){
                if(new_token[j] >= 'A' && new_token[j] <= 'Z'){
                    new_token[j] = new_token[j] - 'A' + 'a';
                    has_upper = true;
                }
            }

            if(has_upper){
                iter = vocab.find(new_token);
                if(iter != vocab.end())
                    return iter->second;
            }
        }
        return -1;
    }
    return iter->second;
}

/* utf-8
  拆分utf8的汉字，把汉字部分独立，ascii部分连续为一个
  for example:
    你jok我good呀  -> ["你", "job", "我", "good", "呀"] */
tuple<vector<string>, vector<tuple<int, int>>> split_words(const string& text){

    // 1字节：0xxxxxxx 
    // 2字节：110xxxxx 10xxxxxx 
    // 3字节：1110xxxx 10xxxxxx 10xxxxxx 
    // 4字节：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
    // 5字节：111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    // 6字节：11111110 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    unsigned char* up = (unsigned char*)text.c_str();
    int offset = 0;
    int length = text.size();
    unsigned char lab_char[] = {
    // 11111110  11111000  11110000  11100000  11000000  01111111
        0xFE,    0xF8,     0xF0,     0xE0,     0xC0,     0x80
    };

    int char_size_table[] = {
        6, 5, 4, 3, 2, 0
    };

    vector<tuple<int, int>> offset_map;
    vector<string> tokens;
    string ascii;
    int state = 0;   // 0 none,  1 wait ascii
    int ascii_start = 0;
    while(offset < length){
        unsigned char token = up[offset];
        int char_size = 1;
        for(int i = 0; i < 6; ++i){
            if(token >= lab_char[i]){
                char_size = char_size_table[i];
                break;
            }
        }

        if(char_size == 0){
            // invalid char
            offset++;
            continue;
        }

        if(offset + char_size > length)
            break;

        auto char_string = text.substr(offset, char_size);
        if(char_size == 1 && token != ' '){
            // ascii 
            if(state == 0){
                ascii = char_string;
                ascii_start = offset;
                state = 1;
            }else if(state == 1){
                ascii += char_string;
            }
        }else{
            if(state == 1){
                tokens.emplace_back(ascii);
                offset_map.emplace_back(ascii_start, offset);
                state = 0;
            }

            if(token != ' '){
                offset_map.emplace_back(offset, offset+char_size);
                tokens.emplace_back(char_string);
            }
        }
        offset += char_size;
    }

    if(state == 1){
        tokens.emplace_back(ascii);
        offset_map.emplace_back(ascii_start, offset);
    }
    return make_tuple(tokens, offset_map);
}

/* 把字符串拆分为词组，汉字单个为一组 */
tuple<vector<string>, vector<tuple<int, int>>> tokenize(const string& text, const unordered_map<string, int>& vocab, int max_length, bool case_to_lower=false){

    vector<tuple<int, int>> offset_map;
    vector<tuple<int, int>> offset_newmap;
    vector<string> tokens;
    vector<string> output;
    auto UNK = "[UNK]";
    tie(tokens, offset_map) = split_words(text);

    for(int itoken = 0; itoken < tokens.size(); ++itoken){
        auto& chars = tokens[itoken];
        int char_start = 0;
        int char_end = 0;
        tie(char_start, char_end) = offset_map[itoken];

        if(chars.size() > max_length){
            output.push_back(UNK);
            offset_newmap.emplace_back(char_start, char_end);
            continue;
        }

        bool is_bad = false;
        int start = 0;
        vector<string> sub_tokens;
        vector<tuple<int, int>> sub_offsetmap;
        while(start < chars.size()){
            int end = chars.size();
            string cur_substr;
            while(start < end){
                auto substr = chars.substr(start, end-start);

                if(case_to_lower){
                    for(int k = 0; k < substr.size(); ++k){
                        auto& c = substr[k];
                        if(c >= 'A' && c <= 'Z')
                            c = c - 'A' + 'a';
                    }
                }

                if(start > 0)
                    substr = "##" + substr;

                auto token_id = find_token(substr, vocab);
                if(token_id != -1){
                    cur_substr = substr;
                    break;
                }
                end -= 1;
            }

            if(cur_substr.empty()){
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            sub_offsetmap.emplace_back(char_start + start, char_start + end);
            start = end;
        }

        if(is_bad){
            output.push_back(UNK);
            offset_newmap.emplace_back(char_start, char_end);
        }else{
            output.insert(output.end(), sub_tokens.begin(), sub_tokens.end());
            offset_newmap.insert(offset_newmap.end(), sub_offsetmap.begin(), sub_offsetmap.end());
        }
    }
    return make_tuple(output, offset_newmap);
}

vector<int> tokens_to_ids(const vector<string>& tokens, const unordered_map<string, int>& vocab){
    vector<int> output(tokens.size());
    for(int i =0 ; i < tokens.size(); ++i)
        output[i] = find_token(tokens[i], vocab);
    return output;
}

tuple<vector<int>, vector<int>, vector<tuple<int, int>>, int> align_and_pad(
    const vector<string>& tokens, vector<tuple<int, int>>& offset_map, int pad_size, 
    const unordered_map<string, int>& vocab
){
    auto CLS = find_token("[CLS]", vocab);
    auto SEP = find_token("[SEP]", vocab);
    vector<int> output = tokens_to_ids(tokens, vocab);
    vector<int> mask(pad_size, 1);
    output.insert(output.begin(), CLS);
    output.insert(output.end(), SEP);
    offset_map.insert(offset_map.begin(), make_tuple(0, 0));
    offset_map.insert(offset_map.end(), make_tuple(0, 0));

    int old_size = output.size();
    output.resize(pad_size);

    if(old_size < pad_size){
        std::fill(output.begin() + old_size, output.end(),   0);
        std::fill(mask.begin()   + old_size, mask.end(),     0);
    }else{
        output.back() = SEP;
        offset_map.back() = make_tuple(0, 0);
    }
    return make_tuple(output, mask, offset_map, old_size);
}

// input_ids, attention_mask, offset_map, word_length
tuple<vector<int>, vector<int>, vector<tuple<int, int>>, int> make_text_data(const string& text, const unordered_map<string, int>& vocab, int max_length){

    vector<string> tokens;
    vector<tuple<int, int>> offset_map;
    tie(tokens, offset_map) = tokenize(text, vocab, max_length);
    return align_and_pad(tokens, offset_map, max_length, vocab);
}


// 上一节的代码
bool build_model(){

    if(exists("ner.trtmodel")){
        printf("ner.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("ner.onnx", 1)){
        printf("Failed to parse ner.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 2;
    printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 30);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < network->getNbInputs(); ++i){
        auto input_tensor = network->getInput(i);
        auto input_dims = input_tensor->getDimensions();
        
        // 配置最小允许batch
        input_dims.d[0] = 1;
        input_dims.d[1] = 512;   // seq length
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = maxBatchSize;
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("ner.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

void inference(){

    TRTLogger logger;
    auto engine_data = load_file("ner.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_seqlength = 512;
    int input_numel = input_batch * input_seqlength;
    auto vocab = load_vocab("../model/vocab.txt");
    int* input_ids_device = nullptr;
    int* input_mask_device = nullptr;
    checkRuntime(cudaMalloc(&input_ids_device, input_numel * sizeof(int)));
    checkRuntime(cudaMalloc(&input_mask_device, input_numel * sizeof(int)));

    ///////////////////////////////////////////////////
    // letter box
    const char* input = "My name is Clara and I live in Berkeley, California.";
    vector<int> input_ids, attention_mask;
    int sentence_length = 0;
    vector<tuple<int, int>> offset_map;
    tie(input_ids, attention_mask, offset_map, sentence_length) = make_text_data(input, vocab, input_seqlength);

    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_ids_device, input_ids.data(), input_numel * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(input_mask_device, attention_mask.data(), input_numel * sizeof(int), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims  = engine->getBindingDimensions(2);
    int output_seqlen = input_seqlength;
    int num_classes   = output_dims.d[2];
    int output_numel  = input_batch * output_seqlen * num_classes;
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_ids_dims = engine->getBindingDimensions(0);
    input_ids_dims.d[0] = input_batch;
    input_ids_dims.d[1] = input_seqlength;
    execution_context->setBindingDimensions(0, input_ids_dims);

    auto input_mask_dims = engine->getBindingDimensions(1);
    input_mask_dims.d[0] = input_batch;
    input_mask_dims.d[1] = input_seqlength;
    execution_context->setBindingDimensions(1, input_mask_dims);

    void* bindings[] = {input_ids_device, input_mask_device, output_data_device};
    bool success      = execution_context->enqueueV2(bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    const char* label_name[] = {
        "O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"
    };
    const char* label_name_human[] = {
        "Miscellaneous entity", "Person's name", "Organization", "Location"
    };
    vector<tuple<string, int, int, int>> output_human;
    vector<int> output_labels(sentence_length);
    int state = 0;
    int cur_start = 0;
    int cur_label = 0;
    for(int i = 0; i < sentence_length; ++i){
        float* pword = output_data_host + i * num_classes;
        int label = std::max_element(pword, pword + num_classes) - pword;
        float confidence = pword[label];
        output_labels[i] = label;

        int start, end;
        tie(start, end) = offset_map[i];
        // printf("label = %d, name = %s, text = ", label, label_name[label]);
        // if(end - start > 0){
        //     printf("[%s]", string(input + start, input + end).c_str());
        // }
        // printf("\n");

        if((label + 1) % 2 == 0){
            // begin
            if(state == 0){
                state = 1;
                cur_start = i;
                cur_label = label;
            }else{
                if(label == cur_label){
                    // pass
                }else{
                    start = get<0>(offset_map[cur_start]);
                    end   = get<1>(offset_map[i-1]);
                    output_human.emplace_back(string(input, start, end - start), start, end, cur_label);

                    state = 1;
                    cur_start = i;
                    cur_label = label;
                }
            }
        }else if((label + 1) % 2 == 1 && label != 0){
            // inside
            if(state == 0){
                state = 1;
                cur_start = i;
                cur_label = label - 1;
            }else{
                if(label - 1 == cur_label){
                    // pass
                }else{
                    start = get<0>(offset_map[cur_start]);
                    end   = get<1>(offset_map[i-1]);
                    output_human.emplace_back(string(input, start, end - start), start, end, cur_label);

                    state = 1;
                    cur_start = i;
                    cur_label = label - 1;
                }
            }
        }else{
            // other
            if(state == 1){
                start = get<0>(offset_map[cur_start]);
                end   = get<1>(offset_map[i-1]);
                output_human.emplace_back(string(input, start, end - start), start, end, cur_label);
                state = 0;
            }
        }
    }

    if(state == 1){
        int start = get<0>(offset_map[cur_start]);
        int end   = get<1>(offset_map[sentence_length-1]);
        output_human.emplace_back(string(input, start, end - start), start, end, cur_label);
        state = 0;
    }

    printf("Input = %s\n", input);
    printf("Predict is: \n");
    for(int i = 0; i < output_human.size(); ++i){
        auto& item = output_human[i];
        string str;
        int start, end, label;
        tie(str, start, end, label) = item;

        printf("text = [%s], start = %d, end = %d, type = %s\n", str.c_str(), start, end, label_name_human[(label - 1) / 2]);
    }

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(output_data_device));
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}