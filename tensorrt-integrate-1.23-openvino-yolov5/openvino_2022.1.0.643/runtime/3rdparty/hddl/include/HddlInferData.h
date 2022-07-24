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

#ifndef __HDDL_INFER_DATA__
#define __HDDL_INFER_DATA__

#include <HddlCommon.h>
#include <HddlBlob.h>

#include <memory>
#include <vector>
#include <functional>

namespace hddl {
class HddlInferDataImpl;

class HDDL_EXPORT_API HddlInferData
{
public:
    typedef std::shared_ptr<HddlInferData> Ptr;

    static HddlInferData::Ptr makeInferData(HddlBlob* in, HddlBlob* out, HddlAuxInfoType auxInfoType = AUX_INFO_NONE);
    ~HddlInferData();

    HddlInferData(const HddlInferData&) = delete;
    HddlInferData& operator=(const HddlInferData&) = delete;

    void           setUserData(const void* data);
    void           setCallback(std::function<void(HddlInferData::Ptr, void*)> callback);

    HddlBlob*      getInputBlob();
    HddlBlob*      getOutputBlob();
    const HddlAuxBlob::Ptr   getAuxInfoBlob();

    HddlStatusCode getInferStatusCode();
    HddlTaskHandle getTaskHandle();

    void*          getUserData();

private:
    friend class HddlClientImpl;
    friend class HddlTask;

    HddlInferData(HddlBlob* in, HddlBlob* out, HddlAuxInfoType auxInfoType);

    HddlInferDataImpl* m_impl;
};
} // namespace hddl

#endif
