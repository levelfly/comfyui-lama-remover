# /nodes/remover.py (最終完整優化版：TensorRT + 動態尺寸 + torch.compile)

import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import gc
from typing import Tuple, List
from contextlib import contextmanager
import math

# 檢查並處理 torch.compile 的可用性
if not hasattr(torch, 'compile'):
    print("⚠️  警告: 您的 PyTorch 版本過低，不支援 torch.compile。建議升級至 2.0 或更高版本以獲得最佳性能。")


    # 定義一個假的裝飾器以確保程式碼在舊版本上也能運行，但沒有性能提升
    def torch_compile_stub(fn, *args, **kwargs):
        return fn


    torch.compile = torch_compile_stub

# 從我們自有的 lama 套件中匯入模型定義
from ..lama import model

# --- 常量定義 ---
PADDING_FACTOR = 32  # 為了數值穩定性，使用較保守的 32 倍填充
DEFAULT_MASK_THRESHOLD = 6
DEFAULT_BLUR_RADIUS = 1
MAX_BATCH_SIZE = 1  # 對於動態尺寸，建議批次為 1 以簡化內存管理

# --- [極限效能融合] ---
try:
    from lama_cpp import _C as custom_cuda_blur

    LAMA_CPP_AVAILABLE = True
    print("✅ 成功匯入自訂 CUDA 模糊核心。")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("⚠️  未找到自訂 CUDA 模糊核心，將使用 PyTorch 原生模糊處理。")


# --- [全局單例模型管理器] ---
class LamaModelManager:
    """全局單例模型管理器，避免重複載入 TensorRT 引擎"""
    _instance = None
    _model = None
    _device = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LamaModelManager, cls).__new__(cls)
        return cls._instance

    @property
    def model(self):
        if not self._is_initialized:
            self._initialize_model()
        return self._model

    @property
    def device(self):
        if not self._is_initialized:
            self._initialize_model()
        return self._device

    def _initialize_model(self):
        if self._is_initialized:
            return
        print("🔄 初始化 LaMa TensorRT 模型...")
        start_time = time.time()
        try:
            self._model = model.BigLama()
            self._device = self._model.device
            self._warmup_model(512, 512)
            init_time = time.time() - start_time
            print(f"✅ 模型初始化完成，耗時: {init_time:.2f}s")
            self._is_initialized = True
        except Exception as e:
            print(f"❌ 模型初始化失敗: {e}")
            raise

    def _warmup_model(self, height, width):
        print(f"🔥 執行模型預熱 (尺寸: {height}x{width})...")
        try:
            dummy_image = torch.randn(1, 3, height, width, device=self._device, dtype=torch.float32)
            dummy_mask = torch.randn(1, 1, height, width, device=self._device, dtype=torch.float32)
            for _ in range(3):
                with torch.no_grad():
                    _ = self._model(dummy_image, dummy_mask)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            print("✅ 模型預熱完成")
        except Exception as e:
            print(f"⚠️  模型預熱失敗: {e}")


model_manager = LamaModelManager()


# --- [高性能模糊處理] ---
class BlurProcessor:
    # 為了 JIT 編譯，將其移出 LamaRemover 類別，使其成為可被編譯的獨立函數
    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _pytorch_gaussian_blur(tensor: torch.Tensor, radius: int) -> torch.Tensor:
        kernel_size = 2 * radius + 1
        sigma = radius / 3.0
        x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()
        gauss = gauss.view(1, 1, 1, -1)
        padding = kernel_size // 2
        tensor = F.conv2d(tensor, gauss, padding=(0, padding), groups=tensor.shape[1])
        tensor = F.conv2d(tensor, gauss.transpose(-1, -2), padding=(padding, 0), groups=tensor.shape[1])
        return tensor

    @staticmethod
    def apply_blur(mask_tensor: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0: return mask_tensor
        if LAMA_CPP_AVAILABLE:
            return custom_cuda_blur.gaussian_blur(mask_tensor, radius)
        else:
            return BlurProcessor._pytorch_gaussian_blur(mask_tensor, radius)


# --- [性能監控] ---
@contextmanager
def performance_monitor(operation_name: str):
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        elapsed_time = end_time - start_time
        memory_delta = (end_memory - start_memory) / 1024 / 1024
        print(f"⏱️  {operation_name}: {elapsed_time:.3f}s, 內存變化: {memory_delta:+.1f}MB")


# --- [優化的主節點類] ---
class LamaRemover:
    """
    極限優化的 LaMa 移除節點 (動態尺寸 + JIT 優化版)
    特點：
    - 單例模型管理，避免重複載入
    - Padding 策略，保持圖像長寬比，提升品質
    - torch.compile JIT 編譯器，加速前後處理
    - 向量化操作，提升計算效率
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {"default": DEFAULT_MASK_THRESHOLD, "min": 0, "max": 255, "step": 1}),
                "gaussblur_radius": ("INT", {"default": DEFAULT_BLUR_RADIUS, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "enable_performance_monitor": ("BOOLEAN", {"default": False, "tooltip": "啟用性能監控（會輕微影響性能）"}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _normalize_tensorrt_output(tensor: torch.Tensor) -> torch.Tensor:
        """向量化並 JIT 編譯的正規化函數"""
        min_val = torch.amin(tensor, dim=(1, 2, 3), keepdim=True)
        max_val = torch.amax(tensor, dim=(1, 2, 3), keepdim=True)
        delta = max_val - min_val + 1e-6
        normalized_tensor = (tensor - min_val) / delta
        is_flat = (max_val - min_val) < 1e-5
        normalized_tensor[is_flat] = 0
        return torch.clamp(normalized_tensor, 0.0, 1.0)

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _pad_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """將張量填充到長寬均為 PADDING_FACTOR 的倍數"""
        b, c, h, w = tensor.shape
        new_h = math.ceil(h / PADDING_FACTOR) * PADDING_FACTOR
        new_w = math.ceil(w / PADDING_FACTOR) * PADDING_FACTOR
        pad_w = new_w - w
        pad_h = new_h - h
        padding = (0, pad_w, 0, pad_h)
        return F.pad(tensor, padding, "constant", 0), (h, w)

    def _prepare_tensors(self, image: torch.Tensor, mask: torch.Tensor, is_image_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        device = model_manager.device
        image_tensor = image.permute(2, 0, 1).unsqueeze(0).to(device)
        padded_image, original_shape = self._pad_tensor(image_tensor)

        if is_image_mask:
            if mask.ndim == 3:
                if mask.shape[2] >= 3:
                    mask = 0.299 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.114 * mask[:, :, 2]
                else:  # Grayscale or other single-channel image
                    mask = mask[:, :, 0]
        elif mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_tensor = mask.unsqueeze(0).unsqueeze(0).to(device)

        if (mask_tensor.shape[2], mask_tensor.shape[3]) != (original_shape[0], original_shape[1]):
            mask_tensor = F.interpolate(mask_tensor, size=original_shape, mode='nearest')

        padded_mask, _ = self._pad_tensor(mask_tensor)
        return padded_image, padded_mask, original_shape

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _postprocess_result(result: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """JIT 編譯的後處理函數，使用 Cropping 策略"""
        h, w = original_shape
        cropped_result = result[:, :, :h, :w]
        return cropped_result.permute(0, 2, 3, 1).cpu()

    def lama_remover(self, images: torch.Tensor, masks: torch.Tensor,
                     mask_threshold: int, gaussblur_radius: int, invert_mask: bool,
                     enable_performance_monitor: bool = False,
                     is_image_mask: bool = False):

        monitor = performance_monitor if enable_performance_monitor else lambda x: contextmanager(lambda: (yield))()

        with monitor("總處理時間"):
            all_results = []
            num_images = images.shape[0]

            for i in range(num_images):
                with monitor(f"圖像 {i + 1}/{num_images}"):
                    try:
                        image = images[i]
                        mask = masks[i]

                        with monitor("數據準備 (Padding)"):
                            padded_image, padded_mask, original_shape = self._prepare_tensors(image, mask, is_image_mask)

                        with monitor("遮罩處理"):
                            if invert_mask: padded_mask = 1.0 - padded_mask
                            if gaussblur_radius > 0: padded_mask = BlurProcessor.apply_blur(padded_mask, gaussblur_radius)
                            threshold = mask_threshold / 255.0
                            padded_mask = (padded_mask > threshold).float()

                        with monitor("TensorRT 推理"):
                            with torch.no_grad():
                                batch_results = model_manager.model(padded_image, padded_mask)
                                if torch.cuda.is_available(): torch.cuda.synchronize()

                        with monitor("結果後處理 (Cropping & Normalize)"):
                            normalized_results = self._normalize_tensorrt_output(batch_results)
                            processed = self._postprocess_result(normalized_results, original_shape)
                            all_results.append(processed)

                    except Exception as e:
                        print(f"❌ 圖像 {i + 1} 處理失敗: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append(images[i:i + 1])

                    if torch.cuda.is_available(): torch.cuda.empty_cache()

        try:
            final_result = torch.cat(all_results, dim=0)
        except Exception as e:
            print(f"❌ 結果合併失敗: {e}")
            final_result = images

        if enable_performance_monitor:
            print(f"🧹 最終 GPU 內存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB")

        return (final_result,)


class LamaRemoverIMG(LamaRemover):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["masks"] = ("IMAGE",)
        return base_inputs

    FUNCTION = "lama_remover_img"

    def lama_remover_img(self, **kwargs):
        return self.lama_remover(is_image_mask=True, **kwargs)


# --- ComfyUI 節點註冊 ---
NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "🚀 Big Lama Remover (JIT 優化版)",
    "LamaRemoverIMG": "🚀 Big Lama Remover IMG (JIT 優化版)"
}