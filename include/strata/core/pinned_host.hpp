// include/strata/core/pinned_host.hpp - large page-locked host blocks on explicit 2 MB pages when asked and
// available.  Huge blocks are always registered Mapped, and require every visible GPU to read them at their host
// address, as with cudaHostAlloc.  A refused huge path takes cudaHostAlloc with the caller's flags unchanged.
#pragma once

#include "strata/platform/huge_pages.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::core {

enum class PinnedBacking { None, CudaHostAlloc, HugeTlb };
constexpr uint64_t kPinnedHugeMin = (uint64_t) 32 << 20;

inline int pinned_huge_env(const char* value) {
    if (value == nullptr || value[0] == '\0') return -1;
    return std::atoi(value) == 0 ? 0 : 1;
}

inline bool pinned_huge_wanted(int env, uint64_t need) {
#ifndef __linux__
    (void) env, (void) need;
    return false;
#else
    if (env == 0 || need == 0) return false;
    if (env == 1) return true;
#ifdef STRATA_USE_HIP
    return false;   // AMD's registered huge pages are untested: explicit opt-in only
#else
    return need >= kPinnedHugeMin && platform::huge_on(-1, platform::huge_pool_2m(), need);
#endif
#endif
}

namespace pinned_host_detail {
struct Registry {
    std::mutex mutex;
    std::unordered_map<const void*, size_t> blocks;
};
// An inline accessor's local static is shared by all host translation units, including CUDA's host code.
inline Registry& registry() {
    static Registry r;
    return r;
}
}  // namespace pinned_host_detail

inline cudaError_t pinned_host_alloc(void** p, size_t bytes, unsigned flags, bool huge,
                                     PinnedBacking* backing = nullptr, int node = -1, int nodes = 1) {
    if (backing != nullptr) *backing = PinnedBacking::None;
    if (huge && p != nullptr && (flags & ~(cudaHostAllocPortable | cudaHostAllocMapped)) == 0) {
        // Preserve cudaHostAlloc's host-address access even when the caller did not ask for Mapped.
        int count = 0;
        bool usable = true;
        if (cudaGetDeviceCount(&count) != cudaSuccess) {
            cudaGetLastError();   // Clear only the failed CUDA call made here.
            usable = false;
        } else if (count == 0) usable = false;
        for (int dev = 0; usable && dev < count; ++dev) {
            int ok = 0;
            if (cudaDeviceGetAttribute(&ok, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess) {
                cudaGetLastError();   // Clear only the failed attribute query.
                usable = false;
            } else if (!ok) usable = false;
        }
        size_t got = 0;
        if (void* m = usable ? platform::huge_alloc(bytes, node, nodes, &got) : nullptr) {
            // Map each huge page before CUDA registers it; the file path already placed the pages.
            for (size_t off = 0; off < got; off += platform::kHugePage2M) ((volatile char*) m)[off] = 0;
            unsigned reg = cudaHostRegisterMapped;
            if ((flags & cudaHostAllocPortable) != 0) reg |= cudaHostRegisterPortable;
            if (cudaHostRegister(m, got, reg) == cudaSuccess) {
                try {
                    auto& r = pinned_host_detail::registry();
                    std::lock_guard<std::mutex> lk(r.mutex);
                    r.blocks.emplace(m, got);
                    *p = m;
                    if (backing != nullptr) *backing = PinnedBacking::HugeTlb;
                    return cudaSuccess;
                } catch (...) {
                    // No untracked registered block: a registry allocation failure takes the fallback too.
                    if (cudaHostUnregister(m) != cudaSuccess)
                        cudaGetLastError();   // Clear only the failed unregister call.
                }
            } else {
                cudaGetLastError();   // Clear only the failed registration.
            }
            platform::huge_free(m, got);
        }
    }
    const cudaError_t err = cudaHostAlloc(p, bytes, flags);
    if (err == cudaSuccess && backing != nullptr) *backing = PinnedBacking::CudaHostAlloc;
    return err;
}

inline cudaError_t pinned_host_free(void* p) {
    if (p == nullptr) return cudaSuccess;
    size_t got = 0;
    {
        auto& r = pinned_host_detail::registry();
        std::lock_guard<std::mutex> lk(r.mutex);
        const auto it = r.blocks.find(p);
        if (it != r.blocks.end()) {
            got = it->second;
            r.blocks.erase(it);
        }
    }
    if (got == 0) return cudaFreeHost(p);
    const cudaError_t err = cudaHostUnregister(p);
    platform::huge_free(p, got);
    return err;
}

inline PinnedBacking pinned_host_backing(const void* p) {
    auto& r = pinned_host_detail::registry();
    std::lock_guard<std::mutex> lk(r.mutex);
    return r.blocks.find(p) != r.blocks.end() ? PinnedBacking::HugeTlb : PinnedBacking::None;
}

}  // namespace strata::core
