#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

namespace {

constexpr int kProbeLength = 1024;

__global__ void fill_kernel(float* data, int count, float value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        data[index] = value;
    }
}

struct ProbeResult
{
    int device_index = 0;
    std::string device_name;
    int cc_major = 0;
    int cc_minor = 0;
    std::uint64_t total_global_mem_bytes = 0;
    bool managed_memory = false;
    bool concurrent_managed_access = false;
    bool unified_addressing = false;
    bool native_kernel_ok = false;
    bool managed_memory_probe_ok = false;
    std::string managed_memory_failure_reason;
    std::string managed_memory_workaround;
};

std::string escape_json(const std::string& value)
{
    std::ostringstream escaped;
    for (unsigned char ch : value)
    {
        switch (ch)
        {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (ch < 0x20)
                {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(ch) << std::dec;
                }
                else
                {
                    escaped << static_cast<char>(ch);
                }
        }
    }
    return escaped.str();
}

void write_probe_result(const std::string& output_path, const ProbeResult& result)
{
    std::ofstream output(output_path, std::ios::out | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("unable to open output path: " + output_path);
    }

    output << "{\n";
    output << "  \"device_index\": " << result.device_index << ",\n";
    output << "  \"device_name\": \"" << escape_json(result.device_name) << "\",\n";
    output << "  \"cc_major\": " << result.cc_major << ",\n";
    output << "  \"cc_minor\": " << result.cc_minor << ",\n";
    output << "  \"total_global_mem_bytes\": " << result.total_global_mem_bytes << ",\n";
    output << "  \"managed_memory\": " << (result.managed_memory ? "true" : "false") << ",\n";
    output << "  \"concurrent_managed_access\": "
           << (result.concurrent_managed_access ? "true" : "false") << ",\n";
    output << "  \"unified_addressing\": "
           << (result.unified_addressing ? "true" : "false") << ",\n";
    output << "  \"native_kernel_ok\": "
           << (result.native_kernel_ok ? "true" : "false") << ",\n";
    output << "  \"managed_memory_probe_ok\": "
           << (result.managed_memory_probe_ok ? "true" : "false");

    if (!result.managed_memory_failure_reason.empty())
    {
        output << ",\n  \"managed_memory_failure_reason\": \""
               << escape_json(result.managed_memory_failure_reason) << "\"";
    }
    if (!result.managed_memory_workaround.empty())
    {
        output << ",\n  \"managed_memory_workaround\": \""
               << escape_json(result.managed_memory_workaround) << "\"";
    }
    output << "\n}\n";
}

std::string describe_error(const char* step, cudaError_t error)
{
    std::ostringstream message;
    message << step << ": " << cudaGetErrorString(error);
    return message.str();
}

bool run_native_probe(ProbeResult& result)
{
    float* device_buffer = nullptr;
    const std::size_t buffer_size = static_cast<std::size_t>(kProbeLength) * sizeof(float);
    cudaError_t error = cudaMalloc(&device_buffer, buffer_size);
    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("cudaMalloc", error);
        return false;
    }

    fill_kernel<<<4, 256>>>(device_buffer, kProbeLength, 1.0f);
    error = cudaGetLastError();
    if (error == cudaSuccess)
    {
        error = cudaDeviceSynchronize();
    }

    float sample_value = 0.0f;
    if (error == cudaSuccess)
    {
        error = cudaMemcpy(&sample_value, device_buffer, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaError_t free_error = cudaFree(device_buffer);
    if (error == cudaSuccess && free_error != cudaSuccess)
    {
        error = free_error;
    }

    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("native_kernel", error);
        return false;
    }

    result.native_kernel_ok = std::fabs(sample_value - 1.0f) < 1e-6f;
    if (!result.native_kernel_ok)
    {
        result.managed_memory_failure_reason = "native kernel verification checksum mismatch";
    }
    return result.native_kernel_ok;
}

bool run_managed_memory_probe(ProbeResult& result)
{
    float* managed_buffer = nullptr;
    const std::size_t buffer_size = static_cast<std::size_t>(kProbeLength) * sizeof(float);
    cudaError_t error = cudaMallocManaged(&managed_buffer, buffer_size);
    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("cudaMallocManaged", error);
        result.managed_memory_workaround =
            "verify Windows driver floor and WSL managed-memory support";
        return false;
    }

    fill_kernel<<<4, 256>>>(managed_buffer, kProbeLength, 2.0f);
    error = cudaGetLastError();
    if (error == cudaSuccess)
    {
        error = cudaDeviceSynchronize();
    }

    float checksum = 0.0f;
    if (error == cudaSuccess)
    {
        for (int index = 0; index < kProbeLength; ++index)
        {
            checksum += managed_buffer[index];
        }
    }

    cudaError_t free_error = cudaFree(managed_buffer);
    if (error == cudaSuccess && free_error != cudaSuccess)
    {
        error = free_error;
    }

    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("managed_memory_probe", error);
        result.managed_memory_workaround =
            "verify Windows driver floor and WSL managed-memory support";
        return false;
    }

    const float expected_checksum = static_cast<float>(kProbeLength) * 2.0f;
    result.managed_memory_probe_ok = std::fabs(checksum - expected_checksum) < 1e-3f;
    if (!result.managed_memory_probe_ok)
    {
        result.managed_memory_failure_reason =
            "managed memory verification checksum mismatch";
        result.managed_memory_workaround =
            "verify Windows driver floor and WSL managed-memory support";
    }
    return result.managed_memory_probe_ok;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: validate_cuda_runtime <output-json>\n";
        return 2;
    }

    const std::string output_path = argv[1];
    ProbeResult result{};

    cudaError_t error = cudaFree(0);
    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("cudaFree(0)", error);
        write_probe_result(output_path, result);
        return 1;
    }

    int device_count = 0;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count < 1)
    {
        result.managed_memory_failure_reason = error == cudaSuccess
            ? "no CUDA devices detected"
            : describe_error("cudaGetDeviceCount", error);
        write_probe_result(output_path, result);
        return 1;
    }

    error = cudaSetDevice(0);
    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("cudaSetDevice", error);
        write_probe_result(output_path, result);
        return 1;
    }

    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess)
    {
        result.managed_memory_failure_reason = describe_error("cudaGetDeviceProperties", error);
        write_probe_result(output_path, result);
        return 1;
    }

    result.device_name = properties.name;
    result.cc_major = properties.major;
    result.cc_minor = properties.minor;
    result.total_global_mem_bytes = properties.totalGlobalMem;
    result.managed_memory = properties.managedMemory != 0;
    result.concurrent_managed_access = properties.concurrentManagedAccess != 0;
    result.unified_addressing = properties.unifiedAddressing != 0;

    if (!run_native_probe(result))
    {
        write_probe_result(output_path, result);
        return 1;
    }

    if (!run_managed_memory_probe(result))
    {
        write_probe_result(output_path, result);
        return 1;
    }

    write_probe_result(output_path, result);
    return 0;
}
