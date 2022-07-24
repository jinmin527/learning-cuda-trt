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

#ifndef __HDDL_API_HDDL_GRAPH_H__
#define __HDDL_API_HDDL_GRAPH_H__

#include <string>
#include <memory>

#include "HddlCommon.h"

namespace hddl {

class HddlGraphImpl;

class HDDL_EXPORT_API HddlGraph
{
public:
    typedef std::shared_ptr<HddlGraph> Ptr;
    ~HddlGraph();

    HddlGraph(const HddlGraph&) = delete;
    HddlGraph& operator=(const HddlGraph&) = delete;

    std::string getName();
    uint64_t getId();
    std::string getPath();

    const void* getData();
    size_t      getDataSize();

    size_t      getInputSize();
    size_t      getOutputSize();
    size_t      getAuxSize(HddlAuxInfoType infoType);

private:
    HddlGraph();

    friend class  HddlClientImpl;
    HddlGraphImpl *m_impl;
};

}
#endif
