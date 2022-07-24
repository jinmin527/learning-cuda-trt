# 使用
1. 运行c++调用lua：`make run -j6`
2. 运行lua调用c++：`make runso -j6`

# 若要单独测试demo.py
1. 设置环境，执行：source `trtpy env-source --print`
2. 然后`cd workspace`后再执行`./pro`即可
3. 或者`cd workspace`后再执行`../lua-build/bin/lua infer.lua`即可

# 若要编译lua
- 执行`bash build-lua.sh`