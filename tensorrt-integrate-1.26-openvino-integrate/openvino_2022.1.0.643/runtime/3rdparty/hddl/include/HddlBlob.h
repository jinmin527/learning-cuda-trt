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

#ifndef HDDL_CLIENT_HDDLBLOB_H
#define HDDL_CLIENT_HDDLBLOB_H

#include "HddlCommon.h"
#include <atomic>
#include <memory>
#include <tuple>

namespace hddl {
class HddlBlobImpl;
class HddlClientImpl;
class HDDL_EXPORT_API HddlBlob {
public:
    HddlBlobImpl* m_impl;

    HddlBlob();
    virtual ~HddlBlob();

    HddlBlob(const HddlBlob&) = delete;
    HddlBlob& operator=(const HddlBlob&) = delete;

    virtual int allocate(size_t size);
    int         reallocate(size_t size);

    void*       getData();
    const void* getData() const;
    size_t      getSize() const;

    /* setRange() sets only the range from offset
     * to (offset+size) of the buffer will be used in inference. */
    void                        setRange(size_t offset, size_t size);
    std::tuple<size_t, size_t>  getRange() const;

protected:
    void setAuxImpl(HddlBlobImpl* impl);
};

class HddlAuxBlobImpl;
class HDDL_EXPORT_API  HddlAuxBlob : public HddlBlob
{
public:
    typedef std::shared_ptr<HddlAuxBlob> Ptr;

    HddlAuxBlob(HddlAuxInfoType auxInfoType);
    ~HddlAuxBlob();

    HddlAuxBlob(const HddlAuxBlob&) = delete;
    HddlAuxBlob& operator=(const HddlAuxBlob&) = delete;

    const void* getAuxData(HddlAuxInfoType auxInfoType, size_t* size) const;
    int allocate(size_t size);

private:
    friend class HddlClientImpl;
    HddlAuxBlobImpl* m_auxImpl;
};


} // namespace hddl

#endif //HDDL_SERVICE_HDDLBUFFER_H
