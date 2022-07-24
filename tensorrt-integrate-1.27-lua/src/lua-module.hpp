#ifndef LUA_MODULE_HPP
#define  LUA_MODULE_HPP

#include <lua.hpp>

#define checkLUA(code, ...)   __checkLUAFunc(L, (code), __FILE__, __LINE__)

bool __checkLUAFunc(lua_State* L, int code, const char* file, int line);

int load_logger_module(lua_State* L, bool global=false);
int load_yolo_module(lua_State* L, bool global=false);

#endif // LUA_MODULE_HPP