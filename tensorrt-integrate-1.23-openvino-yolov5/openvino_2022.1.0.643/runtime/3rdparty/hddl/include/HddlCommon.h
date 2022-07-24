//
// Copyright © 2018-2021 Intel Corporation
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef __INTEL_HDDL_COMMON_EXPORT_FILE_H__
#define __INTEL_HDDL_COMMON_EXPORT_FILE_H__

#include <string>
#include <stddef.h>
#include <stdint.h>

#ifdef WIN32
#ifdef BUILD_HDDL_API
#define HDDL_EXPORT_API __declspec(dllexport)
#else
#define HDDL_EXPORT_API __declspec(dllimport)
#endif
#else
#define HDDL_EXPORT_API  __attribute__((__visibility__("default")))
#endif

namespace hddl {
typedef uint64_t HddlTaskHandle;

typedef enum {
    HDDL_TASK_NOT_FINISHED          =  2,      //Task not finished
    HDDL_TIMEOUT                    =  1,      //Time out
    HDDL_OK                         =  0,

    HDDL_DEVICE_ERROR               = -1,      //Error occurs on myx devices.
    HDDL_CONNECT_ERROR              = -2,      //Error occurs in communication between client and service.
    HDDL_GENERAL_ERROR              = -3,      //General error occurs.
    HDDL_INVALID_PARAM              = -4,      //Invalid parameter provided.
    HDDL_RESOURCE_BUSY              = -5,      //Computer resource is busy or inavaliable.
    HDDL_OPERATION_ERROR            = -6,      //Incorrect operation.
    HDDL_ALLOC_ERROR                = -7,      //Allocate buffer failed.
    HDDL_PERMISSION_DENIED          = -8,      //Permission denied.
    HDDL_NOT_INITIALIZED            = -9,      //Some api used without initialize.
    HDDL_NOT_IMPLEMENTED            = -10,     //Feature not implemented.
} HddlStatusCode;

/* Aux Buffer type*/
typedef enum {
    AUX_INFO_NONE      = 0x00,
    AUX_INFO_TIMETAKEN = 0x01,
    AUX_INFO_THERMAL   = 0x02,
    AUX_INFO_DEBUG     = 0x04,
    AUX_ALL_INFO       = AUX_INFO_TIMETAKEN | AUX_INFO_THERMAL | AUX_INFO_DEBUG
} HddlAuxInfoType;

typedef enum {
    QUERY_TYPE_NONE         = 0x0,
    QUERY_TYPE_PERFORMANCE  = 0x01,
    QUERY_TYPE_DEVICE       = 0x02,
    QUERY_TYPE_GRAPH        = 0x04,
    QUERY_TYPE_CLIENT       = 0x08,
    QUERY_TYPE_ALL          = QUERY_TYPE_PERFORMANCE | QUERY_TYPE_DEVICE | QUERY_TYPE_GRAPH | QUERY_TYPE_CLIENT,
} QueryType;

inline std::string statusCodeToString(HddlStatusCode code)
{
    switch (code) {
        case HDDL_TASK_NOT_FINISHED: return "HDDL_TASK_NOT_FINISHED";
        case HDDL_TIMEOUT:           return "HDDL_TIMEOUT";
        case HDDL_OK:                return "HDDL_OK";
        case HDDL_DEVICE_ERROR:      return "HDDL_DEVICE_ERROR";
        case HDDL_CONNECT_ERROR:     return "HDDL_CONNECT_ERROR";
        case HDDL_GENERAL_ERROR:     return "HDDL_GENERAL_ERROR";
        case HDDL_INVALID_PARAM:     return "HHDDL_INVALID_PARAM";
        case HDDL_RESOURCE_BUSY:     return "HDDL_RESOURCE_BUSY";
        case HDDL_OPERATION_ERROR:   return "HDDL_OPERATION_ERROR";
        case HDDL_ALLOC_ERROR:       return "HDDL_ALLOC_ERROR";
        case HDDL_PERMISSION_DENIED: return "HDDL_PERMISSION_DENIED";
        case HDDL_NOT_INITIALIZED:   return "HDDL_NOT_INITIALIZED";
        case HDDL_NOT_IMPLEMENTED:   return "HDDL_NOT_IMPLEMENTED";
        default:                     return "";
    }

    return "";
}
}

#endif
