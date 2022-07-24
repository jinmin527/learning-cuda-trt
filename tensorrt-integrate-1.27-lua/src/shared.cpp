
#include "lua-module.hpp"

extern "C" int luaopen_yolo(lua_State* L){
	load_logger_module(L);
	return load_yolo_module(L);
}