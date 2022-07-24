// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "hddl_bsl_priv.h"

static BSL_DEVICE_NAME_MATCH s_cfg_device_type_str[BSL_DEVICE_TYPE_MAX] = {
    {"hid-f75114", HID_F75114},
    {"ioexpander", I2C_IOEXPANDER},
    {"mcu", I2C_MCU},
    //{"pch-c246", PCH_C246},
};

BSL_DEVICE_NAME_MATCH* cfg_get_device_type_pair(int idx) {
  return &s_cfg_device_type_str[idx];
}

BslLogLevel_t GlobalLogLevel = BSL_LOG_WARNING;

static const char log_header_template[] = "[%s] [%s] ";
static const char BSL_LOG_LEVEL_SYMBOL[BSL_LOG_DUMMY][3] = {
    "D:",
    "I:",
    "W:",
    "E:",
};

void BSL_print_log(BslLogLevel_t level,  const char* format, ...) {
  if (level < GlobalLogLevel) {
    return;
  }

  va_list args;
  va_start(args, format);

  fprintf(stdout, log_header_template, BSL_LOG_LEVEL_SYMBOL[level], "BSL");
  vfprintf(stdout, format, args);
  fprintf(stdout, "\n");

  va_end(args);
}
