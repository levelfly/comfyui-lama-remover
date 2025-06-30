// /C/gaussian_blur_kernel.cu

#include <torch/extension.h>
#include <vector>
#include <cmath>

// 輔助函式：在 CPU 上計算高斯權重
std::vector<float> create_gaussian_kernel(int radius) {
    float sigma = static_cast<float>(radius) * 0.5f + 0.5f;
    std::vector<float> kernel(radius * 2 + 1);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float val = expf(-(static_cast<float>(i * i)) / (2.0f * sigma * sigma));
        kernel[i + radius] = val;
        sum += val;
    }
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] /= sum;
    }
    return kernel;
}

// 水平方向模糊的 CUDA Kernel
__global__ void horizontal_blur_kernel(const float* input, float* output, int width, int height, const float* kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            int current_x = x + i;
            // 邊界處理: reflect (反射)
            if (current_x < 0) {
                current_x = -current_x;
            } else if (current_x >= width) {
                current_x = 2 * (width - 1) - current_x;
            }
            sum += input[y * width + current_x] * kernel[i + radius];
        }
        output[y * width + x] = sum;
    }
}

// 垂直方向模糊的 CUDA Kernel
__global__ void vertical_blur_kernel(const float* input, float* output, int width, int height, const float* kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            int current_y = y + i;
            // 邊界處理: reflect (反射)
            if (current_y < 0) {
                current_y = -current_y;
            } else if (current_y >= height) {
                current_y = 2 * (height - 1) - current_y;
            }
            sum += input[current_y * width + x] * kernel[i + radius];
        }
        output[y * width + x] = sum;
    }
}

// C++ 函式，負責準備資料並呼叫 CUDA Kernels
torch::Tensor gaussian_blur_cuda_forward(torch::Tensor input, int radius) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (B, C, H, W)");
    TORCH_CHECK(radius > 0, "Radius must be positive");

    // 準備高斯權重
    auto kernel_weights_vec = create_gaussian_kernel(radius);
    auto kernel_tensor = torch::tensor(kernel_weights_vec, torch::dtype(torch::kFloat32).device(input.device()));
    const float* kernel_ptr = kernel_tensor.data_ptr<float>();

    // 準備中間和最終的輸出張量
    auto intermediate = torch::empty_like(input);
    auto output = torch::empty_like(input);

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // 設定 CUDA 執行緒網格
    const dim3 threads(16, 16);
    const dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    // 針對批次中的每個影像和每個通道執行模糊
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const float* input_ptr = input[b][c].data_ptr<float>();
            float* intermediate_ptr = intermediate[b][c].data_ptr<float>();
            float* output_ptr = output[b][c].data_ptr<float>();

            // 執行水平模糊
            horizontal_blur_kernel<<<blocks, threads>>>(input_ptr, intermediate_ptr, W, H, kernel_ptr, radius);
            
            // 執行垂直模糊
            vertical_blur_kernel<<<blocks, threads>>>(intermediate_ptr, output_ptr, W, H, kernel_ptr, radius);
        }
    }
    
    return output;
}