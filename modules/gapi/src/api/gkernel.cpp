// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <iostream> // cerr
#include <functional> // hash
#include <unordered_set>
#include <numeric> // accumulate

#include <ade/util/algorithm.hpp>

#include "logger.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gbackend_priv.hpp"

// GKernelPackage public implementation ////////////////////////////////////////
void cv::gapi::GKernelPackage::remove(const cv::gapi::GBackend& backend)
{
    m_backend_kernels.erase(backend);
}

bool cv::gapi::GKernelPackage::includesAPI(const std::string &id) const
{
    // In current form not very efficient (n * log n)
    auto it = std::find_if(m_backend_kernels.begin(),
                           m_backend_kernels.end(),
                           [&id](const M::value_type &p) {
                               return ade::util::contains(p.second, id);
                           });
    return (it != m_backend_kernels.end());
}

void cv::gapi::GKernelPackage::removeAPI(const std::string &id)
{
    for (auto &bk : m_backend_kernels)
        bk.second.erase(id);
}

std::size_t cv::gapi::GKernelPackage::size() const
{
    return std::accumulate(m_backend_kernels.begin(),
                           m_backend_kernels.end(),
                           static_cast<std::size_t>(0u),
                           [](std::size_t acc, const M::value_type& v) {
                               return acc + v.second.size();
                           });
}

std::vector<std::string>
cv::gapi::GKernelPackage::getConflictKernels(const cv::gapi::GLookupOrder& lookup_order) const
{
    std::vector<std::string> conflict_kernels;
    std::unordered_map<std::string, int> kernels;

    for (const auto &backend : m_backend_kernels)
    {
        for (const auto &kimpl : backend.second)
        {
            kernels[kimpl.first]++;
        }
    }

    for (const auto& k : kernels)
    {
        // Kernel is contained in more than one backend
        if (k.second > 1)
        {
            auto kernel_name = k.first;
            if (lookup_order.empty())
            {
                conflict_kernels.push_back(kernel_name);
            }
            else
            {
                // If lookuporder contains at least one backend
                // with conflicting kernel implementation, the conflict is resolved
                auto conflict_resolved = ade::util::any_of(lookup_order, [this, &kernel_name](const GBackend& backend_from_lookup)
                        {
                            auto it = m_backend_kernels.find(backend_from_lookup);
                            bool backend_in_package = it != m_backend_kernels.end();
                            auto backend_from_package = it->second;
                            return (backend_in_package && ade::util::contains(backend_from_package, kernel_name));
                        });

                if (!conflict_resolved)
                {
                    conflict_kernels.push_back(kernel_name);
                }
            }
        }
    }
    return conflict_kernels;
}

bool cv::gapi::GKernelPackage::includes(const GBackend& backend, const std::string& id)  const
{
    const auto set_iter = m_backend_kernels.find(backend);
    return (set_iter != m_backend_kernels.end())
        ? (ade::util::contains(set_iter->second, id))
        : false;
}

cv::gapi::GKernelPackage cv::gapi::combine(const GKernelPackage  &lhs,
                                           const GKernelPackage  &rhs,
                                           const cv::unite_policy policy)
{

    if (policy == cv::unite_policy::REPLACE)
    {
        // REPLACE policy: if there is a collision, prefer RHS
        // to LHS
        // since RHS package has a precedense, start with its copy
        GKernelPackage result(rhs);
        // now iterate over LHS package and put kernel if and only
        // if there's no such one
        for (const auto &backend : lhs.m_backend_kernels)
        {
            for (const auto &kimpl : backend.second)
            {
                if (!result.includes(backend.first, kimpl.first))
                {
                    result.m_backend_kernels[backend.first].insert(kimpl);
                }
            }
        }
        return result;
    }
    else if (policy == cv::unite_policy::KEEP)
    {
        // KEEP policy: if there is a collision, just keep two versions
        // of a kernel
        GKernelPackage result(lhs);
        for (const auto &p : rhs.m_backend_kernels)
        {
            result.m_backend_kernels[p.first].insert(p.second.begin(),
                                                     p.second.end());
        }
        return result;
    }
    else GAPI_Assert(false);
    return GKernelPackage();
}

std::pair<cv::gapi::GBackend, cv::GKernelImpl>
cv::gapi::GKernelPackage::lookup(const std::string &id,
                                 const GLookupOrder &order) const
{
    if (order.empty())
    {
        // If order is empty, return what comes first
        auto it = std::find_if(m_backend_kernels.begin(),
                               m_backend_kernels.end(),
                               [&id](const M::value_type &p) {
                                   return ade::util::contains(p.second, id);
                               });
        if (it != m_backend_kernels.end())
        {
            // FIXME: Two lookups!
            return std::make_pair(it->first, it->second.find(id)->second);
        }
    }
    else
    {
        // There is order, so:
        // 1. Limit search scope only to specified backends
        //    FIXME: Currently it is not configurable if search can fall-back
        //    to other backends (not listed in order) if kernel hasn't been found
        //    in the look-up list
        // 2. Query backends in the specified order
        for (const auto &selected_backend : order)
        {
            const auto kernels_it = m_backend_kernels.find(selected_backend);
            if (kernels_it == m_backend_kernels.end())
            {
                GAPI_LOG_WARNING(NULL,
                                 "Backend "
                                  << &selected_backend.priv() // FIXME: name instead
                                  << " was listed in lookup list but was not found "
                                     "in the package");
                continue;
            }
            if (ade::util::contains(kernels_it->second, id))
            {
                // FIXME: two lookups!
                return std::make_pair(selected_backend, kernels_it->second.find(id)->second);
            }
        }
    }

    // If reached here, kernel was not found among selected backends.
    util::throw_error(std::logic_error("Kernel " + id + " was not found"));
}

std::vector<cv::gapi::GBackend> cv::gapi::GKernelPackage::backends() const
{
    std::vector<cv::gapi::GBackend> result;
    for (const auto &p : m_backend_kernels) result.emplace_back(p.first);
    return result;
}
