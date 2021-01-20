// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "precomp.hpp"
#include <iostream> // cerr
#include <functional> // hash
#include <numeric> // accumulate

#include <ade/util/algorithm.hpp>

#include "logger.hpp"
#include <opencv2/gapi/gkernel.hpp>

#include "api/gbackend_priv.hpp"

// GKernelPackage public implementation ////////////////////////////////////////
void cv::gapi::GKernelPackage::remove(const cv::gapi::GBackend& backend)
{
    std::vector<std::string> id_deleted_kernels;
    for (const auto& p : m_id_kernels)
    {
        if (p.second.first == backend)
        {
            id_deleted_kernels.push_back(p.first);
        }
    }

    for (const auto& kernel_id : id_deleted_kernels)
    {
        m_id_kernels.erase(kernel_id);
    }
}

bool cv::gapi::GKernelPackage::includesAPI(const std::string &id) const
{
    return ade::util::contains(m_id_kernels, id);
}

void cv::gapi::GKernelPackage::removeAPI(const std::string &id)
{
    m_id_kernels.erase(id);
}

std::size_t cv::gapi::GKernelPackage::size() const
{
    return m_id_kernels.size();
}

const std::vector<cv::GTransform> &cv::gapi::GKernelPackage::get_transformations() const
{
    return m_transformations;
}

cv::gapi::GKernelPackage cv::gapi::combine(const GKernelPackage  &lhs,
                                           const GKernelPackage  &rhs)
{

        // If there is a collision, prefer RHS to LHS
        // since RHS package has a precedense, start with its copy
        GKernelPackage result(rhs);
        // now iterate over LHS package and put kernel if and only
        // if there's no such one
        for (const auto& kernel : lhs.m_id_kernels)
        {
            if (!result.includesAPI(kernel.first))
            {
                result.m_id_kernels.emplace(kernel.first, kernel.second);
            }
        }
        for (const auto &transforms : lhs.m_transformations){
            result.m_transformations.push_back(transforms);
        }
        return result;
}

std::pair<cv::gapi::GBackend, cv::GKernelImpl>
cv::gapi::GKernelPackage::lookup(const std::string &id) const
{
    auto kernel_it = m_id_kernels.find(id);
    if (kernel_it != m_id_kernels.end())
    {
        return kernel_it->second;
    }
    // If reached here, kernel was not found.
    util::throw_error(std::logic_error("Kernel " + id + " was not found"));
}

std::vector<cv::gapi::GBackend> cv::gapi::GKernelPackage::backends() const
{
    using kernel_type = std::pair<std::string, std::pair<cv::gapi::GBackend, cv::GKernelImpl>>;
    std::unordered_set<cv::gapi::GBackend> unique_set;
    ade::util::transform(m_id_kernels, std::inserter(unique_set, unique_set.end()),
                                       [](const kernel_type& k) { return k.second.first; });

    return std::vector<cv::gapi::GBackend>(unique_set.begin(), unique_set.end());
}

cv::gapi::GOutputs cv::gapi::op(const std::string& id, cv::GProtoInputArgs &&ins)
{
    return cv::gapi::GOutputs{id, std::move(ins)};
}

class cv::gapi::GOutputs::Priv
{
public:
    Priv(const std::string& id, cv::GProtoInputArgs &&ins);

    cv::GMat getGMat();

private:
    size_t output = 0;
    std::unique_ptr<cv::GCall> m_call;
};

cv::gapi::GOutputs::Priv::Priv(const std::string& id, cv::GProtoInputArgs &&ins)
{
    cv::GArgs args;
    cv::GKinds kinds;

    for (auto&& proto : ins.m_args)
    {
        switch (proto.index())
        {
            case cv::GProtoArg::index_of<cv::GMat>():
                args.emplace_back(cv::util::get<cv::GMat>(proto));
                kinds.push_back(cv::detail::OpaqueKind::CV_MAT);
                break;

            case cv::GProtoArg::index_of<cv::GScalar>():
                args.emplace_back(cv::util::get<cv::GScalar>(proto));
                kinds.push_back(cv::detail::OpaqueKind::CV_SCALAR);
                break;
            default:
                cv::util::throw_error(std::logic_error("Unsuported input for cv::gapi::op"));
        }
    }
    cv::GKernel k{id, {}, {}, {}, std::move(kinds), {}};
    m_call.reset(new cv::GCall{std::move(k)});
    m_call->setArgs(std::move(args));
}

cv::GMat cv::gapi::GOutputs::Priv::getGMat()
{
    m_call->kernel().outShapes.push_back(cv::GShape::GMAT);
    // ...so _empty_ constructor is passed here.
    m_call->kernel().outCtors.emplace_back(cv::util::monostate{});
    return m_call->yield(output++);
}

cv::gapi::GOutputs::GOutputs(const std::string& id, cv::GProtoInputArgs &&ins) :
    m_priv(new cv::gapi::GOutputs::Priv(id, std::move(ins)))
{
}

cv::GMat cv::gapi::GOutputs::getGMat()
{
    return m_priv->getGMat();
}
