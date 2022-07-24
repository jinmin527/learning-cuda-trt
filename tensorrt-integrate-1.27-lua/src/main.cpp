
#include "lua-module.hpp"
#include <common/ilogger.hpp>

using namespace std;

int main(){

	auto L = luaL_newstate();

	// 启用系统标准库，比如数学运算等
	luaL_openlibs(L);
	load_yolo_module(L);
	load_logger_module(L);

	auto script_code = iLogger::load_text_file("infer.lua");
	if(!checkLUA(luaL_loadstring(L, script_code.c_str())))
		return -1;
	
    if(!checkLUA(lua_pcall(L, 0, 1, 0)))
		return -1;
	
	INFO("Done.");
	return 0;
}