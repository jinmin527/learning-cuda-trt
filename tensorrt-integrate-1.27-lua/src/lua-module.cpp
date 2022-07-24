
#include "lua-module.hpp"
#include <lualib.h>

#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include "builder/trt_builder.hpp"
#include "app_yolo/yolo.hpp"

using namespace std;

bool __checkLUAFunc(lua_State* L, int code, const char* file, int line){
    if(code != 0){                                      
        auto msg = lua_tostring(L, -1);                  
        luaL_traceback(L,L,msg,1);                      
        iLogger::__log_func(file, line, iLogger::LogLevel::Error, "Error, code = %d, message: %s", code, lua_tostring(L,-1));      
        return false;                                   
    }           
    return true;                                        
}

static int compile(
    lua_State* L
){
	luaL_argcheck(L, lua_istable(L, -1), 0, "args must table");
	luaL_argcheck(L, lua_getfield(L, -1, "max_batch_size") == LUA_TNUMBER, 0, "max_batch_size must be integer");
	int max_batch_size = lua_tonumber(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "source") == LUA_TSTRING, 0, "source must be string");
	string source = lua_tostring(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "output") == LUA_TSTRING, 0, "output must be string");
	string output = lua_tostring(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "fp16") == LUA_TBOOLEAN, 0, "fp16 must be bool");
	bool fp16 = lua_toboolean(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "device_id") == LUA_TNUMBER, 0, "device_id must be integer");
	int device_id = lua_tonumber(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "max_workspace_size") == LUA_TNUMBER, 0, "max_workspace_size must be integer");
	size_t max_workspace_size = lua_tonumber(L, -1);  lua_pop(L, 1);

	// pop argument dict
	lua_pop(L, 1);

	auto result = TRT::compile(
        fp16 ? TRT::Mode::FP16 : TRT::Mode::FP32,
        max_batch_size, source, output, {}, nullptr, "", "", max_workspace_size
    );
	lua_pushboolean(L, result);
	return 1;
}

static int infer_forward(lua_State* L){

	luaL_argcheck(L, lua_istable(L, -1), 0, "args must table");
	luaL_argcheck(L, lua_istable(L, -2), 0, "object must table");

	if(!lua_getfield(L, -2, "__ptr")){
		lua_pushboolean(L, false);
		INFOE("infer no __ptr attribute");
		return 1;
	}
	auto infer = *(std::shared_ptr<Yolo::Infer>*)lua_touserdata(L, -1);
    lua_pop(L, 1);
	
	luaL_argcheck(L, lua_getfield(L, -1, "input") == LUA_TSTRING, 0, "input must be string");
	string input = lua_tostring(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "output") == LUA_TSTRING, 0, "output must be string");
	string output = lua_tostring(L, -1);  lua_pop(L, 1);

	auto image = cv::imread(input);
	if(image.empty()){
		lua_pushboolean(L, false);
		INFOE("input is empty: %s", input.c_str());
		return 1;
	}

	auto boxes = infer->commit(image).get();
	for(auto& box : boxes)
		cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0,255, 0), 2);
	
	lua_pushboolean(L, cv::imwrite(output, image));
	return 1;
}

static int create_infer(
    lua_State* L
){
	luaL_argcheck(L, lua_istable(L, -1), 0, "args must table");
	luaL_argcheck(L, lua_getfield(L, -1, "file") == LUA_TSTRING, 0, "file must be string");
	string file = lua_tostring(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "gpuid") == LUA_TNUMBER, 0, "gpuid must be integer");
	int gpuid = lua_tonumber(L, -1);  lua_pop(L, 1);

	luaL_argcheck(L, lua_getfield(L, -1, "threshold") == LUA_TNUMBER, 0, "threshold must be float");
	float threshold = lua_tonumber(L, -1);  lua_pop(L, 1);

	auto ins = new shared_ptr<Yolo::Infer>(Yolo::create_infer(file, Yolo::Type::V5, gpuid, threshold));
	
	lua_newtable(L);
	lua_pushlightuserdata(L, ins);
    lua_setfield(L, -2, "__ptr");
	lua_pushcfunction(L, infer_forward);
    lua_setfield(L, -2, "forward");
	return 1;
}

static int exists(lua_State* L){

	luaL_argcheck(L, lua_type(L, -1) == LUA_TSTRING, 0, "file must be string");
	auto file = lua_tostring(L, -1); lua_pop(L, 1);
	auto result = iLogger::exists(file);

	lua_pushboolean(L, result);
	return 1;
}

static const luaL_Reg yolo_module_register[] = {
	{"compile", compile},
	{"create_infer", create_infer},
	{"exists", exists},
	{nullptr, nullptr}
};

static int new_yolo_lua(lua_State* L){

	luaL_newlib(L, yolo_module_register);
	return 1;
}

int load_yolo_module(lua_State* L, bool global){
    luaL_requiref(L, "yolo", new_yolo_lua, global);
    return 0;
}

////////////////////////////////////////////////////////////////

static int logger_info(lua_State* L){

	auto msg = lua_tostring(L, -1);
	INFO("%s", msg); lua_pop(L, 1);
	return 0;
}

static int logger_error(lua_State* L){

	auto msg = lua_tostring(L, -1);
	INFOE("%s", msg); lua_pop(L, 1);
	return 0;
}

static int logger_warning(lua_State* L){

	auto msg = lua_tostring(L, -1);
	INFOW("%s", msg); lua_pop(L, 1);
	return 0;
}

static const luaL_Reg logger_module_register[] = {
	{"info", logger_info},
	{"warning", logger_warning},
	{"error", logger_error},
	{nullptr, nullptr}
};

static int new_logger_lua(lua_State* L){
	luaL_newlib(L, logger_module_register);
	return 1;
}

int load_logger_module(lua_State* L, bool global){
    luaL_requiref(L, "logger", new_logger_lua, global);
    return 0;
}