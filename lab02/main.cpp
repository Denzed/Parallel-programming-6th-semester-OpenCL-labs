#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION CL_HPP_TARGET_OPENCL_VERSION

#include <CL/cl2.hpp>
#include "cl.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

const int BLOCKSIZE = 256;

size_t divide_rounded_up(size_t divisible, size_t divisor) {
    return (divisible + divisor - 1) / divisor;
}

size_t upscale_to_divisible(size_t divisible, size_t divisor) {
    return divide_rounded_up(divisible, divisor) * divisor;
}

struct exclusive_scan_data {
    cl::Context context;
    cl::KernelFunctor<cl::Buffer,int,cl::Buffer,cl::LocalSpaceArg> scan_kernel;
    cl::KernelFunctor<cl::Buffer,int,cl::Buffer> expand_kernel;
    cl::CommandQueue queue;
    
    std::vector<float> scan(std::vector<float> &in) {
        size_t n = in.size();
        std::vector<float> out(n);
        
        inner_scan(in, out, n);

        if (n > BLOCKSIZE) {
            std::vector<float> block_sums;
            for (size_t i = BLOCKSIZE - 1; i < n; i += BLOCKSIZE) {
                block_sums.push_back(out[i] + in[i]);
            }
            std::vector<float> block_scans = scan(block_sums);

            inner_expand(block_scans, block_scans.size(), out, n);
        }

        return out;
    }
private:
    void inner_scan(std::vector<float> &in, std::vector<float> &out, size_t n) {
        // initialize buffers
        cl::Buffer input(context, CL_MEM_READ_ONLY,  sizeof(float) * n);
        cl::Buffer output(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        // load to GPU
        queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(float) * n, in.data());
        
        // enqueue args
        cl::EnqueueArgs eargs(
            queue, 
            cl::NullRange, 
            cl::NDRange(upscale_to_divisible(n, BLOCKSIZE)), 
            cl::NDRange(BLOCKSIZE)
        );

        scan_kernel(eargs, input, n, output, {sizeof(float) * BLOCKSIZE});

        // load from GPU
        queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(float) * n, out.data());
    }

    void inner_expand(
        std::vector<float> &partial_scans, size_t m, 
        std::vector<float> &block_scans, size_t n
    ) {
        // initialize buffers
        cl::Buffer input(context, CL_MEM_READ_ONLY,  sizeof(float) * m);
        cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(float) * n);

        // load to GPU
        queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(float) * m, partial_scans.data());
        queue.enqueueWriteBuffer(output, CL_TRUE, 0, sizeof(float) * n, block_scans.data());
        
        // enqueue args
        cl::EnqueueArgs eargs(
            queue, 
            cl::NullRange, 
            cl::NDRange(upscale_to_divisible(n, BLOCKSIZE)), 
            cl::NDRange(BLOCKSIZE)
        );

        expand_kernel(eargs, input, n, output);

        // load from GPU
        queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(float) * n, block_scans.data());
    }
};

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // create context
    cl::Context context(devices);

    // load opencl source
    std::ifstream cl_file("exclusive_scan.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, cl_string);

    // read data
    std::ifstream input_file("input.txt");
    size_t N;
    input_file >> N;
    
    std::vector<float> a;
    a.resize(N);

    for (auto &elem: a) {
        input_file >> elem;
    }

    // create program
    cl::Program program(context, source);

    try {
        // compile opencl source
        program.build(devices);   
    } catch (cl::Error e) {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }

        return 1;
    }

    // initialize kernels and command queue
    cl::KernelFunctor<cl::Buffer,int,cl::Buffer,cl::LocalSpaceArg> scan_kernel(program, "scan_blelloch");

    cl::KernelFunctor<cl::Buffer,int,cl::Buffer> expand_kernel(program, "expand");

    exclusive_scan_data data = {
        context,
        scan_kernel,
        expand_kernel,
        cl::CommandQueue(context, devices[0])
    };

    // process 
    std::vector<float> output = data.scan(a);

    // output result
    std::ofstream output_file("output.txt");
    for (auto &elem: output) {
        output_file << std::setprecision(3) << std::fixed << elem << " ";
    }
    output_file << std::endl;

    return 0;
}