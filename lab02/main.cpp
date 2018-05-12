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
    return (divisible + divisor - 1) / divisor
}

size_t upscale_to_divisible(size_t divisible, size_t divisor) {
    return divide_rounded_up(divisible, divisor) * divisor;
}

struct inclusive_scan_data {
    cl::Context context;
    cl::KernelFunctor<cl::Buffer,int,cl::Buffer,cl::Buffer> scan_kernel;
    cl::KernelFunctor<cl::Buffer,int,cl::Buffer> reduce_kernel;
    cl::CommandQueue queue;
    
    std::vector<double> scan(std::vector<double> &in) {
        size_t n = in.size();
        std::vector<double> out(n);
        
        inner_scan(in, out, n);

        if (n > BLOCKSIZE) {
            std::vector<double> block_sums;
            for (size_t i = BLOCKSIZE - 1; i < n; i += BLOCKSIZE) {
                block_sums.push_back(out[i]);
            }
            std::vector<double> block_scans = scan(block_sums);

            inner_reduce(block_scans, block_scans.size(), out, n);
        }

        return out;
    }
private:
    void inner_scan(vector<double> &in, vector<double> &out, size_t n) {
        // initialize buffers
        cl::Buffer input(context, CL_MEM_READ_ONLY,  sizeof(double) * n);
        cl::Buffer output(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

        // load to GPU
        queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(double) * n, in.data());
        
        // enqueue args
        cl::EnqueueArgs eargs(
            queue, 
            cl::NullRange, 
            cl::NDRange(upscale_to_divisible(n, BLOCKSIZE)), 
            cl::NDRange(BLOCKSIZE)
        );

        scan_kernel(eargs, input, n, output, cl::__local(sizeof(double) * BLOCKSIZE));

        // load from GPU
        queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(double) * n, out.data());
    }

    void inner_reduce(
        vector<double> &partial_scans, size_t m, 
        vector<double> &block_scans, size_t n
    ) {
        // initialize buffers
        cl::Buffer input(context, CL_MEM_READ_ONLY,  sizeof(double) * m);
        cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(double) * n);

        // load to GPU
        queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(double) * m, partial_scans.data());
        queue.enqueueWriteBuffer(output, CL_TRUE, 0, sizeof(double) * n, block_scans.data());
        
        // enqueue args
        cl::EnqueueArgs eargs(
            queue, 
            cl::NullRange, 
            cl::NDRange(upscale_to_divisible(n, BLOCKSIZE)), 
            cl::NDRange(BLOCKSIZE)
        );

        reduce_kernel(eargs, input, n, output);

        // load from GPU
        queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(double) * n, block_scans.data());
    }
};

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // load opencl source
        std::ifstream cl_file("inclusive_scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        // read data
        std::ifstream input_file("input.txt");
        size_t N;
        input_file >> N;
        
        std::vector<double> a;
        a.resize(N);

        for (auto &elem: a) {
            input_file >> elem;
        }

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // initialize kernels and command queue
        inclusive_scan_data data = {
            context,
            cl::KernelFunctor<cl::Buffer,int,cl::Buffer,cl::Buffer>(program, "scan_blelloch"),
            cl::KernelFunctor<cl::Buffer,int,cl::Buffer>(program, "reduce"),
            cl::CommandQueue queue(context, devices[0])
        };

        // process 
        std::vector<double> output = data.scan(a);

        // output result
        std::ofstream output_file("output.txt");
        for (auto &elem: output) {
            output_file << elem << " ";
        }
        output_file << std::endl;
    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}