// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <hddl-bsl.h>
#include <stdio.h>
#include <stdlib.h>

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"

HddlController_t m_hddl_controller[BSL_DEVICE_TYPE_MAX];
static BslDeviceType m_bsl_device = DEVICE_INVALID_TYPE;
static bool initialized = false;

static void bsl_fill_all_callback_functions();

static BSL_STATUS hddl_bsl_init_with_config(char* cfg_path);
static BSL_STATUS bsl_set_and_config_device(BslDeviceType dev_type, CFG_HANDLER config);
static BSL_STATUS bsl_init_by_auto_scan();

static BSL_STATUS hddl_bsl_destroy();

#ifndef WIN32
__attribute__((destructor))
#endif
void libbsl_destroy()
{
  hddl_bsl_destroy();
}

BSL_STATUS hddl_bsl_init() {
  if (initialized) {
    return BSL_SUCCESS;
  }

  bsl_fill_all_callback_functions();

  char path[MAX_PATH_LENGTH];
  BSL_STATUS status = cfg_get_path(path, sizeof(path));

  if (status == BSL_SUCCESS) {
    status = hddl_bsl_init_with_config(path);
    BslLog(BSL_LOG_INFO, "Config file detected at %s", path);
    goto exit_init;
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_TOO_LONG) {
    BslLog(BSL_LOG_ERROR, "${HDDL_INSTALL_DIR} is too long");
    BslLog(BSL_LOG_ERROR, "Please reduce the length of this value then retry");
    return status;
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_NOT_DIR) {
    BslLog(BSL_LOG_ERROR, "${HDDL_INSTALL_DIR}=%s is not a valid dir", path);
    BslLog(BSL_LOG_ERROR, "Please check the correctness of that path");
    return status;
  }

  if (status == BSL_ERROR_HDDL_INSTALL_DIR_NOT_PROVIDED) {
    BslLog(BSL_LOG_INFO, "${HDDL_INSTALL_DIR} not provided");
  }

  if (status == BSL_ERROR_CFG_OPEN_FAILED) {
    BslLog(BSL_LOG_INFO, "Config file decided based on ${HDDL_INSTALL_DIR} is %s", path);
    BslLog(BSL_LOG_INFO, "Config file open failed due to non-existing or permissions");
  }

  BslLog(BSL_LOG_INFO, "Trying to init with auto-scan");
  status = bsl_init_by_auto_scan();

exit_init:
  if (status == BSL_SUCCESS) {
    initialized = true;
  }
  return status;
}

static BSL_STATUS hddl_bsl_init_with_config(char* cfg_path) {
  CFG_HANDLER config = CFG_HANDLER_DEFAULT_VALUE;
  BSL_STATUS status;
  status = cfg_open(cfg_path, &config);
  hddl_set_log_level(cfg_get_log_level(config));
  for (int i = 0; i < BSL_DEVICE_TYPE_MAX; i++) {
    BSL_DEVICE_NAME_MATCH* name_pair = cfg_get_device_type_pair(i);
    CFG_HANDLER device_cfg = cfg_get_field(config, name_pair->device_name);
    if (device_cfg == NULL) {
      continue;
    }

    bool enabled = cfg_type_is_enabled(device_cfg);
    if (!enabled) {
      BslLog(BSL_LOG_INFO, "%s is disabled by config, skipping", name_pair->device_name);
      continue;
    }

    status = bsl_set_and_config_device(name_pair->device_type, device_cfg);
    if (status == BSL_SUCCESS) {
      cfg_close(config);
      return status;
    }
    BslLog(BSL_LOG_ERROR, "%s init returned status ", name_pair->device_name);
    hddl_get_error_string(status);
  }

  if (cfg_get_autoscan_switch(config)) {
    status = bsl_init_by_auto_scan();
  } else {
    BslLog(BSL_LOG_INFO, "Auto-scan is disabled by config, aborting");
  }

  cfg_close(config);
  return status;
}

void bsl_fill_all_callback_functions() {
  mcu_init(&m_hddl_controller[I2C_MCU]);
  ioexpander_init(&m_hddl_controller[I2C_IOEXPANDER]);
  hid_f75114_init(&m_hddl_controller[HID_F75114]);
  // c246_init(&m_hddl_controller[PCH_C246]);
}

BSL_STATUS bsl_set_and_config_device(BslDeviceType dev_type, CFG_HANDLER config) {
  BSL_STATUS status;

  device_config_t device_config_callback = m_hddl_controller[dev_type].device_config;
  if (device_config_callback == NULL) {
    return BSL_ERROR_CALLBACK_NOT_FOUND;
  }
  status = device_config_callback(config);
  if (BSL_SUCCESS != status) {
    return status;
  }
  status = hddl_set_device(dev_type);
  if (BSL_SUCCESS != status) {
    return status;
  }
  device_init_t device_init_callback = m_hddl_controller[dev_type].device_init;
  if (device_init_callback == NULL) {
    return BSL_ERROR_CALLBACK_NOT_FOUND;
  }
  return device_init_callback();
}

BSL_STATUS bsl_init_by_auto_scan() {
  BSL_STATUS status;
  int device_count = 0;

  BslLog(BSL_LOG_INFO, "Performing auto-scan");
  for (int i = 0; i < BSL_DEVICE_TYPE_MAX; i++) {
    BslDeviceType device_type = cfg_get_device_type_pair(i)->device_type;
    if (!is_valid_device_type(device_type)) {
      continue;
    }
    status = m_hddl_controller[device_type].device_scan(&device_count);
    if (status != BSL_SUCCESS) {
      BslLog(BSL_LOG_DEBUG, "scan device type %d failed with %d", i, status);
      continue;
    }

    if (device_count > 0) {
      BslLog(BSL_LOG_INFO, "Found %d devices", device_count);
      BslDeviceType dev_type = device_type;
      hddl_set_device(dev_type);
      status = m_hddl_controller[dev_type].device_init();
      return BSL_SUCCESS;
    }
  }
  BslLog(BSL_LOG_ERROR, "No device found");

  return BSL_ERROR_NO_DEVICE_FOUND;
}

BSL_STATUS hddl_check_device() {
  bsl_fill_all_callback_functions();

  BSL_STATUS status;
  int device_count = 0;

  BslLog(BSL_LOG_INFO, "Performing auto-scan");
  for (int i = 0; i < BSL_DEVICE_TYPE_MAX; i++) {
    BslDeviceType device_type = cfg_get_device_type_pair(i)->device_type;
    if (!is_valid_device_type(device_type)) {
      continue;
    }
    status = m_hddl_controller[device_type].device_scan(&device_count);
    if (status != BSL_SUCCESS) {
      BslLog(BSL_LOG_DEBUG, "scan device type %d failed with %d", i, status);
      continue;
    }

    if (device_count > 0) {
      BslLog(BSL_LOG_INFO, "Found %d devices", device_count);
      return BSL_SUCCESS;
    }
  }

  return BSL_ERROR_NO_DEVICE_FOUND;
}

static BSL_STATUS hddl_bsl_destroy() {
  if (!initialized) {
    return BSL_SUCCESS;
  }

  BslDeviceType dev_type = hddl_get_device();
  BSL_STATUS status = BSL_SUCCESS;
  if (is_valid_device_type(dev_type)) {
    status = m_hddl_controller[dev_type].device_destroy();
  }
  return status;
}

BSL_STATUS hddl_reset(int device_addr) {
  BSL_STATUS status = hddl_bsl_init();
  if (status != BSL_SUCCESS) {
    return status;
  }

  BslLog(BSL_LOG_INFO, "Reset device address: %d with device type %d", device_addr, m_bsl_device);
  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  device_reset_t reset_single_device_callback = m_hddl_controller[m_bsl_device].device_reset;
  if (reset_single_device_callback) {
    return reset_single_device_callback(device_addr);
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BSL_STATUS hddl_reset_all() {
  BSL_STATUS status = hddl_bsl_init();
  if (status != BSL_SUCCESS) {
    return status;
  }

  BslLog(BSL_LOG_INFO, "Reset all devices with device type %d", m_bsl_device);

  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  device_reset_all_t reset_all_callback = m_hddl_controller[m_bsl_device].device_reset_all;
  if (reset_all_callback) {
    return reset_all_callback();
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BSL_STATUS hddl_discard(int device_addr) {
  BSL_STATUS status = hddl_bsl_init();
  if (status != BSL_SUCCESS) {
    return status;
  }

  if (!is_valid_device_type(m_bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  BslLog(BSL_LOG_INFO, "Discard device: %d", device_addr);
  device_discard_t device_discard_callback = m_hddl_controller[m_bsl_device].device_discard;
  if (device_discard_callback) {
    return device_discard_callback(device_addr);
  }
  return BSL_ERROR_CALLBACK_NOT_FOUND;
}

BslDeviceType hddl_get_device() {
  BSL_STATUS status = hddl_bsl_init();
  if (status != BSL_SUCCESS) {
    return status;
  }

  return m_bsl_device;
}

BSL_STATUS hddl_set_device(BslDeviceType bsl_device) {
  BslLog(BSL_LOG_INFO, "hddl_set_device bsl_device=%d", bsl_device);
  if (!is_valid_device_type(bsl_device)) {
    return BSL_ERROR_INVALID_DEVICE_TYPE;
  }

  int device_count = 0;
  BSL_STATUS status = m_hddl_controller[bsl_device].device_get_device_num(&device_count);
  if (0 == device_count || status != BSL_SUCCESS) {
    // if it is not 0, it should be scan already in hddl_bsl_init
    // if it is 0, it may not do a scan in hddl_bsl_init
    status = m_hddl_controller[bsl_device].device_scan(&device_count);
    if (status != BSL_SUCCESS) {
      return status;
    }
    device_init_t device_init_callback = m_hddl_controller[bsl_device].device_init;
    if (device_init_callback == NULL) {
      return BSL_ERROR_CALLBACK_NOT_FOUND;
    }
    status = device_init_callback();
  }
  if (status != BSL_SUCCESS) {
    return status;
  }
  if (0 == device_count) {
    return BSL_ERROR_NO_DEVICE_FOUND;
  }

  m_bsl_device = bsl_device;
  initialized = true;
  return BSL_SUCCESS;
}

BSL_STATUS hddl_set_log_level(BslLogLevel_t level) {
  GlobalLogLevel = level;
  return BSL_SUCCESS;
}

bool is_valid_device_type(BslDeviceType bsl_device) {
  return (bsl_device >= BSL_DEVICE_TYPE_START) && (bsl_device < BSL_DEVICE_TYPE_MAX);
}
