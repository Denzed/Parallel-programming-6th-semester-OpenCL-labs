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

const int BLOCKSIZE = 16;

void read_float_matrix(std::ifstream &input_file, float *matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        input_file >> matrix[i];
    }
}

size_t upscale_to_divisible(size_t divisible, size_t divisor) {
    size_t mod = divisible % divisor;
    if (mod == 0) {
        return divisible;
    } else {
        return divisible + divisor - mod;
    }
}

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

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("matrix_conv.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        // read data
        std::ifstream input_file("input.txt");
        size_t N, M;
        input_file >> N >> M;
        
        float a[N * N];
        float b[M * M];
        float c[N * N];
          
        read_float_matrix(input_file, a, N * N);
        read_float_matrix(input_file, b, M * M);
        memset(c, 0, sizeof(c));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(
            devices, 
            (
                "-D BLOCKSIZE=" + std::to_string(BLOCKSIZE) + " "
                "-D M=" + std::to_string(M)
            ).c_str()
        );

        // create message 
        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(a));
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(b));
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(c));

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(a), a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(b), b);
        
        // load named kernel from opencl source
        cl::KernelFunctor<
               cl::Buffer,
               int,
               cl::Buffer,
               int,
               cl::Buffer
            > matrix_conv(program, "matrix_conv");
        cl::EnqueueArgs matrix_conv_eargs(
            queue, 
            cl::NullRange, 
            cl::NDRange(
                upscale_to_divisible(N, BLOCKSIZE), 
                upscale_to_divisible(N, BLOCKSIZE)
            ), 
            cl::NDRange(BLOCKSIZE, BLOCKSIZE)
        );
        matrix_conv(matrix_conv_eargs, dev_a, N, dev_b, M, dev_c);

        // copy from GPU to CPU
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(c), c);

#ifdef DEBUG
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                std::cout << a[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                std::cout << b[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "finished" << std::endl;
#endif

        // output result
        std::ofstream output_file("output.txt");
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                output_file << std::setprecision(2) << std::fixed << c[idx] << " ";
            }
            output_file << std::endl;
        }
    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}