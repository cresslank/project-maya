// tests/pinned_host_test.cu - the ordinary pinned path and explicit 2 MB registration, through a real GPU copy.
// No model; the huge-page part uses the host's configured pool, or checks the ordinary allocation fallback.
#include "strata/core/pinned_host.hpp"

#include <cstdio>
#include <cstring>

namespace c = strata::core;
namespace p = strata::platform;

static int fails = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                   \
        }                                                              \
    } while (0)

static bool cuda_ok(cudaError_t err, const char* call) {
    if (err == cudaSuccess) return true;
    std::printf("FAIL %s: %s\n", call, cudaGetErrorString(err));
    ++fails;
    return false;
}

#ifdef __linux__
static bool kernel_2m(const void* p) {
    std::FILE* f = std::fopen("/proc/self/smaps", "r");
    if (f == nullptr) return false;
    const auto addr = (uintptr_t) p;
    bool in_mapping = false, found = false;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0, kb = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) == 2) in_mapping = addr >= lo && addr < hi;
        else if (in_mapping && std::sscanf(line, "KernelPageSize: %llu kB", &kb) == 1) {
            found = kb == 2048;
            break;
        }
    }
    std::fclose(f);
    return found;
}
#endif

static unsigned char pattern(size_t i) { return (unsigned char) ((i * 37 + i / 251) & 255); }

int main() {
    CHECK(c::pinned_huge_env(nullptr) == -1 && c::pinned_huge_env("") == -1);
    CHECK(c::pinned_huge_env("0") == 0 && c::pinned_huge_env("00") == 0);
    CHECK(c::pinned_huge_env("off") == 0);
    CHECK(c::pinned_huge_env("1") == 1 && c::pinned_huge_env("2") == 1 && c::pinned_huge_env("-1") == 1);
    CHECK(!c::pinned_huge_wanted(0, 64 * 1024 * 1024));
    CHECK(!c::pinned_huge_wanted(0, 0));
#ifdef __linux__
    CHECK(c::pinned_huge_wanted(1, 1));
    if (p::huge_pool_2m().total() == 0) CHECK(!c::pinned_huge_wanted(-1, 64 << 20));
#else
    CHECK(!c::pinned_huge_wanted(1, 1));
    CHECK(!c::pinned_huge_wanted(1, 64 << 20));
#endif
    CHECK(!c::pinned_huge_wanted(1, 0));
    CHECK(!c::pinned_huge_wanted(-1, 0));
    CHECK(!c::pinned_huge_wanted(-1, c::kPinnedHugeMin - 1));
    CHECK(c::pinned_host_free(nullptr) == cudaSuccess);
    CHECK(c::pinned_host_backing(nullptr) == c::PinnedBacking::None);

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        cudaGetLastError();
        std::printf("skipped: no CUDA GPU (pinned setting rules checked)\n");
        return fails == 0 ? 0 : 1;
    }
    {
        void* host = nullptr;
        c::PinnedBacking backing = c::PinnedBacking::None;
        if (cuda_ok(c::pinned_host_alloc(&host, 8 * 1024 * 1024, cudaHostAllocPortable, false, &backing),
                    "ordinary pinned_host_alloc")) {
            CHECK(backing == c::PinnedBacking::CudaHostAlloc);
            CHECK(c::pinned_host_backing(host) == c::PinnedBacking::None);
            CHECK(c::pinned_host_free(host) == cudaSuccess);
            CHECK(c::pinned_host_backing(host) == c::PinnedBacking::None);
            std::printf("pinned host: ordinary 8 MiB Portable allocation and free\n");
        }
    }
    {   // Plain cudaHostAlloc callers may also use the shared free function.
        void* host = nullptr;
        if (cuda_ok(cudaHostAlloc(&host, 8 * 1024 * 1024, cudaHostAllocPortable), "plain cudaHostAlloc"))
            CHECK(c::pinned_host_free(host) == cudaSuccess);
    }
    constexpr size_t bytes = (size_t) 64 << 20;
    bool mapped_usable = true;
    for (int dev = 0; dev < devices; ++dev) {
        int ok = 0;
        if (!cuda_ok(cudaDeviceGetAttribute(&ok, cudaDevAttrCanUseHostPointerForRegisteredMem, dev),
                     "registered host pointer attribute") || !ok) mapped_usable = false;
    }
    const bool pool_holds = p::huge_pool_2m().total() >= bytes;
    void* host = nullptr;
    c::PinnedBacking backing = c::PinnedBacking::None;
    if (cuda_ok(c::pinned_host_alloc(&host, bytes, cudaHostAllocPortable, true, &backing),
                "huge pinned_host_alloc")) {
        const bool expect_huge = pool_holds && mapped_usable;
        CHECK(backing == (expect_huge ? c::PinnedBacking::HugeTlb : c::PinnedBacking::CudaHostAlloc));
        CHECK(c::pinned_host_backing(host) == (expect_huge ? c::PinnedBacking::HugeTlb : c::PinnedBacking::None));
        std::printf("pinned host: 64 MiB Portable %s (pool %s, registered host pointers %s)\n",
                    backing == c::PinnedBacking::HugeTlb ? "HugeTlb" : "CudaHostAlloc fallback",
                    pool_holds ? "holds it" : "too small", mapped_usable ? "supported" : "unsupported");
        void* alias = nullptr;
        CHECK(cuda_ok(cudaHostGetDevicePointer(&alias, host, 0), "cudaHostGetDevicePointer"));
        if (backing == c::PinnedBacking::HugeTlb) CHECK(alias == host);
#ifdef __linux__
        if (backing == c::PinnedBacking::HugeTlb) CHECK(kernel_2m(host));
#endif
        auto* data = (unsigned char*) host;
        for (size_t i = 0; i < bytes; ++i) data[i] = pattern(i);
        void* device = nullptr;
        if (cuda_ok(cudaMalloc(&device, bytes), "cudaMalloc")) {
            if (cuda_ok(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "host to device")) {
                std::memset(host, 0, bytes);
                if (cuda_ok(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "device to host")) {
                    bool match = true;
                    for (size_t i = 0; i < bytes; ++i) {
                        if (data[i] != pattern(i)) {
                            std::printf("pinned host: pattern differs at byte %zu\n", i);
                            match = false;
                            break;
                        }
                    }
                    CHECK(match);
                    if (match) std::printf("pinned host: 64 MiB pattern round trip matched\n");
                }
            }
            CHECK(cudaFree(device) == cudaSuccess);
        }
        CHECK(c::pinned_host_free(host) == cudaSuccess);
        CHECK(c::pinned_host_backing(host) == c::PinnedBacking::None);
    }
    {   // An explicit Mapped request keeps the same huge-page and host-address guarantees.
        host = nullptr;
        backing = c::PinnedBacking::None;
        if (cuda_ok(c::pinned_host_alloc(&host, bytes, cudaHostAllocPortable | cudaHostAllocMapped, true, &backing),
                    "mapped huge pinned_host_alloc")) {
            CHECK(backing == (pool_holds && mapped_usable ? c::PinnedBacking::HugeTlb : c::PinnedBacking::CudaHostAlloc));
            void* alias = nullptr;
            CHECK(cuda_ok(cudaHostGetDevicePointer(&alias, host, 0), "mapped cudaHostGetDevicePointer"));
            if (backing == c::PinnedBacking::HugeTlb) CHECK(alias == host);
            CHECK(c::pinned_host_free(host) == cudaSuccess);
        }
    }
    if (fails == 0) std::printf("pinned host: ok\n");
    return fails == 0 ? 0 : 1;
}
