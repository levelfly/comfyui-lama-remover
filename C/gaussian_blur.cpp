// /C/gaussian_blur.cpp

#include <torch/extension.h>

// 宣告我們將在 .cu 檔案中實現的 CUDA 函式原型
torch::Tensor gaussian_blur_cuda_forward(torch::Tensor input, int radius);

// 將 C++ 函式綁定到 Python 模組
// TORCH_EXTENSION_NAME 是一個特殊的宏，它會被 setup.py 中定義的名稱取代
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
      "gaussian_blur", // 在 Python 中呼叫時的函式名稱
      &gaussian_blur_cuda_forward, // 對應的 C++ 函式指標
      "A custom and fast Gaussian Blur implementation in CUDA" // 函式說明文件
    );
}