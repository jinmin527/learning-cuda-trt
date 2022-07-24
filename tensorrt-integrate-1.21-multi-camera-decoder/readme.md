# 硬件解码
- 增加NVDEC和ffmpeg的配置
- 软解码和硬解码，分别消耗cpu和gpu资源。在多路，大分辨率下体现明显
- 硬件解码和推理可以允许跨显卡
- 理解并善于利用的时候，他才可能发挥最大的效果

# 使用
1. 为nvcuvid创建软链接，这个库随显卡驱动发布
    - `bash link-cuvid.sh`
2. 测试python接口，`make run -j64`
3. 测试C++接口，多摄像头混合解码，显存占用最低
    - `make runpro -j64`
    - 核心思想是把NALU交替放入解码器进行解码，最后通过timestamp进行区分

# 如果要在目录下执行
- 请执行
    ```bash
    source `trtpy env-source --print`
    ```
    把环境source到当前，就可以进行`./pro yolo`了