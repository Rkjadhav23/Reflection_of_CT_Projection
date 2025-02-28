#include <OpenCL/cl-patched.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <cnpy.h>
#include <fstream>
#include <sstream>

// Function to print a matrix (for debugging purposes)
void print_matrix(const std::vector<float>& matrix, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

// CPU reference implementation
std::vector<float> get_fanbeam_reflection_cpu(const std::vector<float>& sinogram, int height, int halfwidth,
                                              float SDD = 1.0856, float pixelSize = 0.002884778911564626, float axis_pixelposition = 0.5) {
    //int width = halfwidth * 2;
    std::vector<float> output(height * halfwidth, 0.0f);

    for (int u_right = 0; u_right < halfwidth; ++u_right) {
        int u_left = halfwidth - u_right - 1;
        for (int i = 0; i < height; ++i) {
            float angle = i * 2 * M_PI / height;
            float b_inright = angle - atan2((u_right - axis_pixelposition) * pixelSize, SDD);

            // Wrap angle to [0, 2π]
            if (b_inright < 0) b_inright += 2 * M_PI;

            // Nearest neighbor interpolation
            int idx = static_cast<int>(round(b_inright * height / (2 * M_PI))) % height;
            output[i * halfwidth + u_right] = sinogram[idx * halfwidth + u_left];
        }
    }

    return output;
}

// GPU implementation using OpenCL buffers
double get_fanbeam_reflection_gpu_buffer(const float* sinogram, const float* right_sinogram, int height, int halfwidth,
                                         float SDD = 1.0856, float pixelSize = 0.002884778911564626, float axis_pixelposition = 0.5) {
    //int width = halfwidth * 2;

    // Set up OpenCL context, command queue, and program
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load OpenCL kernel code
    std::ifstream kernelFile("kernels.cl");
    std::string kernelCode((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    cl::Program program(context, kernelCode);
    program.build({device});

    // Allocate buffers on the GPU
    cl::Buffer d_sinogram(context, CL_MEM_READ_ONLY, sizeof(float) * height * halfwidth);
    cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * height * halfwidth);

    // Copy sinogram data to GPU
    queue.enqueueWriteBuffer(d_sinogram, CL_TRUE, 0, sizeof(float) * height * halfwidth, sinogram);

    // Set kernel arguments
    cl::Kernel kernel(program, "fanbeam_reflection_buffer");
    kernel.setArg(0, d_sinogram);
    kernel.setArg(1, d_output);
    kernel.setArg(2, height);
    kernel.setArg(3, halfwidth);
    kernel.setArg(4, SDD);
    kernel.setArg(5, pixelSize);
    kernel.setArg(6, axis_pixelposition);

    // Execute kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(height, halfwidth));

    // Read output back to host
    std::vector<float> output(height * halfwidth);
    queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * height * halfwidth, output.data());

    // Compute MSE
    float mse = 0.0f;
    for (int i = 0; i < height * halfwidth; ++i) {
        float diff = output[i] - right_sinogram[i];
        mse += diff * diff;
    }
    mse /= (height * halfwidth);

    return static_cast<double>(mse);
}

// GPU implementation using OpenCL images
double get_fanbeam_reflection_gpu_image(const float* sinogram, const float* right_sinogram, int height, int halfwidth,
                                        float SDD = 1.0856, float pixelSize = 0.002884778911564626, float axis_pixelposition = 0.5) {
    //int width = halfwidth * 2;

    // Set up OpenCL context, command queue, and program
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load OpenCL kernel code
    std::ifstream kernelFile("kernels.cl");
    std::string kernelCode((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    cl::Program program(context, kernelCode);
    program.build({device});

    // Create OpenCL image for sinogram
    cl::Image2D d_sinogram(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), halfwidth, height);
    cl::Image2D d_output(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), halfwidth, height);

    // Copy sinogram data to GPU image
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = halfwidth;
    region[1] = height;
    region[2] = 1;
    queue.enqueueWriteImage(d_sinogram, CL_TRUE, origin, region, 0, 0, sinogram);

    // Set kernel arguments
    cl::Kernel kernel(program, "fanbeam_reflection_image");
    kernel.setArg(0, d_sinogram);
    kernel.setArg(1, d_output);
    kernel.setArg(2, height);
    kernel.setArg(3, halfwidth);
    kernel.setArg(4, SDD);
    kernel.setArg(5, pixelSize);
    kernel.setArg(6, axis_pixelposition);

    // Execute kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(height, halfwidth));

    // Read output back to host
    std::vector<float> output(height * halfwidth);
    queue.enqueueReadImage(d_output, CL_TRUE, origin, region, 0, 0, output.data());

    // Compute MSE
    float mse = 0.0f;
    for (int i = 0; i < height * halfwidth; ++i) {
        float diff = output[i] - right_sinogram[i];
        mse += diff * diff;
    }
    mse /= (height * halfwidth);

    return static_cast<double>(mse);
}

int main() {
    //auto start = std::chrono::high_resolution_clock::now();

    // Load sinogram data
cnpy::NpyArray arr = cnpy::npy_load("/zhome/jadhavrk/Downloads/Sinogram_0.5.npy");
    if (arr.shape.size() != 2 || arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected a 2D array of floats." << std::endl;
        return 1;
    }

    float* data = arr.data<float>();
    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];

    // Split sinogram into left and right halves
    std::vector<float> half_matrix(rows * (cols / 2));
    std::vector<float> right_matrix(rows * (cols / 2));
    for (size_t i = 0; i < rows; ++i) {
        std::copy(data + i * cols, data + i * cols + cols / 2, half_matrix.begin() + i * (cols / 2));
        std::copy(data + i * cols + cols / 2, data + i * cols + cols, right_matrix.begin() + i * (cols / 2));
    }

    // CPU reference implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_output = get_fanbeam_reflection_cpu(half_matrix, rows, cols / 2);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_elapsed.count() << " seconds" << std::endl;

    // GPU buffer implementation
    auto gpu_buffer_start = std::chrono::high_resolution_clock::now();
    double gpu_buffer_error = get_fanbeam_reflection_gpu_buffer(half_matrix.data(), right_matrix.data(), rows, cols / 2);
    auto gpu_buffer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_buffer_elapsed = gpu_buffer_end - gpu_buffer_start;
    std::cout << "GPU (buffer) execution time: " << gpu_buffer_elapsed.count() << " seconds" << std::endl;
    std::cout << "GPU (buffer) error: " << gpu_buffer_error << std::endl;

    // GPU image implementation
    auto gpu_image_start = std::chrono::high_resolution_clock::now();
    double gpu_image_error = get_fanbeam_reflection_gpu_image(half_matrix.data(), right_matrix.data(), rows, cols / 2);
    auto gpu_image_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_image_elapsed = gpu_image_end - gpu_image_start;
    std::cout << "GPU (image) execution time: " << gpu_image_elapsed.count() << " seconds" << std::endl;
    std::cout << "GPU (image) error: " << gpu_image_error << std::endl;

    // Calculate speedup
    std::cout << "Speedup (buffer): " << cpu_elapsed.count() / gpu_buffer_elapsed.count() << "x" << std::endl;
    std::cout << "Speedup (image): " << cpu_elapsed.count() / gpu_image_elapsed.count() << "x" << std::endl;

    // Print system information
    std::cout << "CPU: Intel Core i7-9750H" << std::endl;
    std::cout << "GPU: NVIDIA GeForce GTX 1650" << std::endl;
    std::cout << "Computer: Lab PC" << std::endl;

    return 0;
}
