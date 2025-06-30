# /nodes/remover.py (動態尺寸優化版：單例模型 + Padding/Cropping + 性能監控)

import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import gc
from typing import Tuple, Optional, List
from contextlib import contextmanager
import math

# 從我們自有的 lama 套件中匯入模型定義
from ..lama import model

# --- 常量定義 (MODIFIED) ---
# MODEL_INPUT_SIZE 已被移除，因為我們不再使用固定尺寸
PADDING_FACTOR = 32  # LaMa 模型通常要求輸入尺寸是 8 的倍數
DEFAULT_MASK_THRESHOLD = 128
DEFAULT_BLUR_RADIUS = 10
MAX_BATCH_SIZE = 1  # 對於動態尺寸，建議批次為 1 以簡化內存管理

# --- [極限效能融合] ---
try:
    from lama_cpp import _C as custom_cuda_blur

    LAMA_CPP_AVAILABLE = True
    print("✅ 成功匯入自訂 CUDA 模糊核心。已啟用極限效能模式。")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("⚠️  未找到自訂 CUDA 模糊核心。將使用 PyTorch 原生模糊處理。")
    from PIL import ImageFilter


# --- [全局單例模型管理器] ---
class LamaModelManager:
    """
    全局單例模型管理器，避免重複載入 TensorRT 引擎
    """
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
        """惰性初始化模型"""
        if self._is_initialized:
            return

        print("🔄 初始化 LaMa TensorRT 模型...")
        start_time = time.time()

        try:
            self._model = model.BigLama()
            self._device = self._model.device

            # 對於動態尺寸引擎，使用一個常見尺寸進行預熱
            self._warmup_model(512, 512)

            init_time = time.time() - start_time
            print(f"✅ 模型初始化完成，耗時: {init_time:.2f}s")

            self._is_initialized = True

        except Exception as e:
            print(f"❌ 模型初始化失敗: {e}")
            raise

    def _warmup_model(self, height, width):
        """預熱模型以優化首次推理"""
        print(f"🔥 執行模型預熱 (尺寸: {height}x{width})...")
        try:
            # 創建虛擬輸入進行預熱
            dummy_image = torch.randn(1, 3, height, width,
                                      device=self._model.device, dtype=torch.float32)
            dummy_mask = torch.randn(1, 1, height, width,
                                     device=self._model.device, dtype=torch.float32)

            # 執行幾次預熱推理
            for _ in range(3):
                with torch.no_grad():
                    _ = self._model(dummy_image, dummy_mask)
                    torch.cuda.synchronize()

            print("✅ 模型預熱完成")

        except Exception as e:
            print(f"⚠️  模型預熱失敗: {e}")


# 全局模型管理器實例
model_manager = LamaModelManager()


# --- [內存池管理器] ---
# 備註：對於動態尺寸，傳統的內存池效果有限，因為形狀不斷變化。此處保留結構但影響降低。
class TensorPool:
    """
    張量內存池，重用張量以減少內存分配開銷
    """

    def __init__(self, device):
        self.device = device
        self.pools = {}

    def get_tensor(self, shape, dtype=torch.float32):
        key = (tuple(shape), dtype)
        if key not in self.pools: self.pools[key] = []
        pool = self.pools[key]
        return pool.pop().zero_() if pool else torch.zeros(shape, dtype=dtype, device=self.device)

    def return_tensor(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.pools: self.pools[key] = []
        if len(self.pools[key]) < 10: self.pools[key].append(tensor.detach())


# --- [高性能模糊處理] ---
class BlurProcessor:
    @staticmethod
    def apply_blur(mask_tensor: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0: return mask_tensor
        if LAMA_CPP_AVAILABLE:
            return custom_cuda_blur.gaussian_blur(mask_tensor, radius)
        else:
            return BlurProcessor._pytorch_gaussian_blur(mask_tensor, radius)

    @staticmethod
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
    極限優化的 LaMa 移除節點 (動態尺寸版)
    特點：
    - 單例模型管理，避免重複載入
    - Padding 策略，保持圖像長寬比，提升品質
    - Cropping 後處理，還原原始尺寸
    - 性能監控，實時追蹤效能
    """

    def __init__(self):
        self.tensor_pool = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {"default": DEFAULT_MASK_THRESHOLD, "min": 0, "max": 255, "step": 1}),
                "gaussblur_radius": ("INT", {"default": DEFAULT_BLUR_RADIUS, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": MAX_BATCH_SIZE, "step": 1, "tooltip": "對於動態尺寸, 建議設為 1"}),
                "enable_performance_monitor": ("BOOLEAN", {"default": False, "tooltip": "啟用性能監控（會輕微影響性能）"}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def _initialize_tensor_pool(self):
        if self.tensor_pool is None:
            self.tensor_pool = TensorPool(model_manager.device)

    def _normalize_tensorrt_output(self, tensor: torch.Tensor) -> torch.Tensor:
        # (此函數無需修改，歸一化邏輯保持不變)
        normalized_tensors = []
        for i in range(tensor.shape[0]):
            sample = tensor[i:i + 1]
            min_val, max_val = torch.min(sample), torch.max(sample)
            if max_val > min_val:
                normalized_sample = (sample - min_val) / (max_val - min_val)
            else:
                normalized_sample = torch.zeros_like(sample)
            normalized_sample = torch.clamp(normalized_sample, 0.0, 1.0)
            normalized_tensors.append(normalized_sample)
        return torch.cat(normalized_tensors, dim=0)

    def _validate_inputs(self, images: torch.Tensor, masks: torch.Tensor, is_image_mask: bool = False) -> bool:
        # (此函數無需修改，驗證邏輯保持不變)
        if images.shape[0] != masks.shape[0]: return False
        if len(images.shape) != 4: return False
        if is_image_mask and len(masks.shape) != 4: return False
        if not is_image_mask and len(masks.shape) != 3: return False
        return True

    # --- NEW: Padding 輔助函數 ---
    def _pad_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        將張量填充到長寬均為 PADDING_FACTOR 的倍數
        返回填充後的張量和原始的填充邊界
        """
        b, c, h, w = tensor.shape

        # 計算新的目標長寬
        new_h = math.ceil(h / PADDING_FACTOR) * PADDING_FACTOR
        new_w = math.ceil(w / PADDING_FACTOR) * PADDING_FACTOR

        # 計算需要填充的量 (右邊和下邊)
        pad_w = new_w - w
        pad_h = new_h - h

        # 使用 F.pad 進行填充
        # (pad_left, pad_right, pad_top, pad_bottom)
        padding = (0, pad_w, 0, pad_h)
        padded_tensor = F.pad(tensor, padding, "constant", 0)

        print(f"🖼️  Padding: 原始尺寸 ({h}, {w}) -> 填充後尺寸 ({new_h}, {new_w})")

        return padded_tensor, (h, w)

    # --- REWRITTEN: 資料準備，從 Resize 改為 Padding ---
    def _prepare_tensors(self, image: torch.Tensor, mask: torch.Tensor, is_image_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        為單個圖像和遮罩準備張量，使用 Padding 策略
        """
        device = model_manager.device

        # 1. 處理圖像
        image_tensor = image.permute(2, 0, 1).unsqueeze(0).to(device)  # HWC -> BCHW
        padded_image, original_shape = self._pad_tensor(image_tensor)

        # 2. 處理遮罩 (統一格式)
        if is_image_mask:
            if mask.ndim == 3 and mask.shape[2] > 1:
                if mask.shape[2] >= 3:
                    mask = 0.299 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.114 * mask[:, :, 2]
                else:
                    mask = mask[:, :, 0]
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask[:, :, 0]
        elif mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_tensor = mask.unsqueeze(0).unsqueeze(0).to(device)  # HW -> BCHW

        # 3. 將遮罩縮放並填充到與圖像相同的尺寸
        # 首先，將遮罩縮放至原始圖像大小（如果它們不匹配）
        if (mask_tensor.shape[2], mask_tensor.shape[3]) != (original_shape[0], original_shape[1]):
            mask_tensor = transforms.functional.resize(
                mask_tensor,
                (original_shape[0], original_shape[1]),
                interpolation=transforms.InterpolationMode.NEAREST
            )
        # 然後，使用與圖像完全相同的 Padding
        padded_mask, _ = self._pad_tensor(mask_tensor)

        return padded_image, padded_mask, original_shape

    # --- REWRITTEN: 結果後處理，從 Resize 改為 Cropping ---
    def _postprocess_result(self, result: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """
        後處理結果張量，使用 Cropping 策略
        """
        h, w = original_shape

        # 從左上角裁剪回原始尺寸
        cropped_result = result[:, :, :h, :w]

        # 轉換為 ComfyUI 格式 (BCHW -> BHWC) 和 CPU
        result_comfy = cropped_result.permute(0, 2, 3, 1).cpu()

        return result_comfy

    def lama_remover(self, images: torch.Tensor, masks: torch.Tensor,
                     mask_threshold: int, gaussblur_radius: int, invert_mask: bool,
                     batch_size: int = 1, enable_performance_monitor: bool = False,
                     is_image_mask: bool = False):
        if not self._validate_inputs(images, masks, is_image_mask):
            return (images,)

        monitor = performance_monitor if enable_performance_monitor else lambda x: contextmanager(lambda: (yield))()

        with monitor("總處理時間"):
            all_results = []
            num_images = images.shape[0]

            # --- MODIFIED: 簡化迴圈以適應動態尺寸 ---
            # 動態尺寸使得批次內尺寸統一變得很複雜，逐張處理是最穩定可靠的方式。
            # 原有的 batch_size 參數保留，但內部邏輯以單張處理為主。
            for i in range(num_images):
                with monitor(f"圖像 {i + 1}/{num_images}"):
                    try:
                        image = images[i:i + 1]  # 取單張圖
                        mask = masks[i:i + 1]  # 取對應遮罩

                        # 準備數據 (Padding)
                        with monitor("數據準備 (Padding)"):
                            # 注意：is_image_mask 判斷需要原始的張量維度
                            is_img_mask_flag = is_image_mask and len(masks.shape) == 4
                            padded_image, padded_mask, original_shape = self._prepare_tensors(image[0], mask[0], is_img_mask_flag)

                        # 遮罩預處理
                        with monitor("遮罩處理"):
                            if invert_mask:
                                padded_mask = 1.0 - padded_mask
                            if gaussblur_radius > 0:
                                padded_mask = BlurProcessor.apply_blur(padded_mask, gaussblur_radius)
                            threshold = mask_threshold / 255.0
                            padded_mask = (padded_mask > threshold).float()

                        # 模型推理
                        with monitor("TensorRT 推理"):
                            with torch.no_grad():
                                batch_results = model_manager.model(padded_image, padded_mask)
                                torch.cuda.synchronize()

                        # 結果後處理
                        with monitor("結果後處理 (Cropping & Normalize)"):
                            normalized_results = self._normalize_tensorrt_output(batch_results)
                            processed = self._postprocess_result(normalized_results, original_shape)
                            all_results.append(processed)

                    except Exception as e:
                        print(f"❌ 圖像 {i + 1} 處理失敗: {e}")
                        import traceback
                        traceback.print_exc()
                        # 備援：返回原始圖像
                        all_results.append(images[i:i + 1])

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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

    def lama_remover_img(self, images: torch.Tensor, masks: torch.Tensor,
                         mask_threshold: int, gaussblur_radius: int, invert_mask: bool,
                         batch_size: int = 1, enable_performance_monitor: bool = False):
        return self.lama_remover(
            images=images,
            masks=masks,
            mask_threshold=mask_threshold,
            gaussblur_radius=gaussblur_radius,
            invert_mask=invert_mask,
            batch_size=batch_size,
            enable_performance_monitor=enable_performance_monitor,
            is_image_mask=True
        )


def get_model_info():
    try:
        return model_manager.model.get_engine_info()
    except Exception as e:
        return {"error": str(e)}


def cleanup_resources():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    print("🧹 資源清理完成")


# --- ComfyUI 節點註冊 (MODIFIED) ---
NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "🚀 Big Lama Remover (動態尺寸版)",
    "LamaRemoverIMG": "🚀 Big Lama Remover IMG (動態尺寸版)"
}