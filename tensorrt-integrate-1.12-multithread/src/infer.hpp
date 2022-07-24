#ifndef INFER_HPP
#define INFER_HPP

#include <string>
#include <future>
#include <memory>

/////////////////////////////////////////////////////////////////////////////////////////
// 封装接口类
class Infer{
public:
    virtual std::shared_future<std::string> commit(const std::string& input) = 0;
};

std::shared_ptr<Infer> create_infer(const std::string& file);

#endif // INFER_HPP