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

#ifndef HDDL_SERVICE_HDDLVM_H
#define HDDL_SERVICE_HDDLVM_H

#include <map>
#include <string>
namespace hddl {
struct GroupDeviceParams {
    std::string groupTag;
    uint32_t deviceNum;
    int subclass;
    std::map<std::string, uint32_t> graphTagMap;
    uint32_t streamDeviceNum;
    uint32_t bypassDeviceNum;
    uint32_t sgadDeviceNum;
    bool useSgadByDefault;
    int deviceScheduleInterval;
    int maxGraphPerDevice;
    int maxCycleSwitchOut;
    int maxTaskNumberSwitchOut;
};

class HddlClient;
class HDDL_EXPORT_API HddlVm {
public:
    HddlVm(const char* name);
    HddlVm(std::string& string);
    ~HddlVm();

    HddlStatusCode groupDevice(const GroupDeviceParams& params);
    HddlStatusCode ungroupDevice(const std::string& groupTag);
private:
    HddlClient* m_client;
};


} // namespace hddl

#endif //HDDL_SERVICE_HDDLVM_H
