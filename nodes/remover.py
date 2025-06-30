# /nodes/remover.py (極限優化版本：單例模型 + 批處理 + 內存池 + 性能監控)

import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import gc
from typing import Tuple, Optional, List
from contextlib import contextmanager

# 從我們自有的 lama 套件中匯入模型定義
from ..lama import model

# --- 常量定義 ---
MODEL_INPUT_SIZE = 512  # LaMa 模型的標準輸入尺寸
DEFAULT_MASK_THRESHOLD = 128
DEFAULT_BLUR_RADIUS = 10
MAX_BATCH_SIZE = 4  # RTX 3090 的最佳批處理大小

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

            # 預熱模型以優化首次推理速度
            self._warmup_model()

            init_time = time.time() - start_time
            print(f"✅ 模型初始化完成，耗時: {init_time:.2f}s")

            self._is_initialized = True

        except Exception as e:
            print(f"❌ 模型初始化失敗: {e}")
            raise

    def _warmup_model(self):
        """預熱模型以優化首次推理"""
        print("🔥 執行模型預熱...")
        try:
            # 創建虛擬輸入進行預熱
            dummy_image = torch.randn(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                                      device=self._model.device, dtype=torch.float32)
            dummy_mask = torch.randn(1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
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
class TensorPool:
    """
    張量內存池，重用張量以減少內存分配開銷
    """

    def __init__(self, device):
        self.device = device
        self.pools = {}  # 按形狀分組的張量池

    def get_tensor(self, shape, dtype=torch.float32):
        """獲取指定形狀的張量"""
        key = (tuple(shape), dtype)

        if key not in self.pools:
            self.pools[key] = []

        pool = self.pools[key]

        if pool:
            tensor = pool.pop()
            tensor.zero_()  # 清零重用
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=self.device)

    def return_tensor(self, tensor):
        """歸還張量到池中"""
        key = (tuple(tensor.shape), tensor.dtype)

        if key not in self.pools:
            self.pools[key] = []

        # 限制池大小避免內存洩漏
        if len(self.pools[key]) < 10:
            self.pools[key].append(tensor.detach())


# --- [高性能模糊處理] ---
class BlurProcessor:
    """
    高性能模糊處理器，支持批處理和 CUDA 加速
    """

    @staticmethod
    def apply_blur(mask_tensor: torch.Tensor, radius: int) -> torch.Tensor:
        """
        應用高斯模糊，自動選擇最佳實現
        """
        if radius <= 0:
            return mask_tensor

        if LAMA_CPP_AVAILABLE:
            # 使用自定義 CUDA 核心（最快）
            return custom_cuda_blur.gaussian_blur(mask_tensor, radius)
        else:
            # 使用 PyTorch 原生實現（較快）
            return BlurProcessor._pytorch_gaussian_blur(mask_tensor, radius)

    @staticmethod
    def _pytorch_gaussian_blur(tensor: torch.Tensor, radius: int) -> torch.Tensor:
        """
        使用 PyTorch 實現的批處理高斯模糊
        """
        # 計算高斯核大小
        kernel_size = 2 * radius + 1

        # 創建高斯核
        sigma = radius / 3.0
        x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
        x = x - kernel_size // 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()

        # 重塑為 2D 核
        gauss = gauss.view(1, 1, 1, -1)

        # 應用可分離的高斯模糊（水平 + 垂直）
        # 水平模糊
        padding = kernel_size // 2
        tensor = F.conv2d(tensor, gauss, padding=(0, padding), groups=tensor.shape[1])
        # 垂直模糊
        tensor = F.conv2d(tensor, gauss.transpose(-1, -2), padding=(padding, 0), groups=tensor.shape[1])

        return tensor


# --- [性能監控] ---
@contextmanager
def performance_monitor(operation_name: str):
    """性能監控上下文管理器"""
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        elapsed_time = end_time - start_time
        memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB

        print(f"⏱️  {operation_name}: {elapsed_time:.3f}s, 內存變化: {memory_delta:+.1f}MB")


# --- [優化的主節點類] ---
class LamaRemover:
    """
    極限優化的 LaMa 移除節點
    特點：
    - 單例模型管理，避免重複載入
    - 智能批處理，充分利用 RTX 3090
    - 內存池管理，減少分配開銷
    - 性能監控，實時追蹤效能
    """

    def __init__(self):
        self.tensor_pool = None  # 惰性初始化

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {
                    "default": DEFAULT_MASK_THRESHOLD,
                    "min": 0, "max": 255, "step": 1
                }),
                "gaussblur_radius": ("INT", {
                    "default": DEFAULT_BLUR_RADIUS,
                    "min": 0, "max": 50, "step": 1
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": MAX_BATCH_SIZE, "step": 1,
                    "tooltip": "批處理大小，RTX 3090 建議 2-4"
                }),
                "enable_performance_monitor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "啟用性能監控（會輕微影響性能）"
                }),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def _initialize_tensor_pool(self):
        """惰性初始化張量池"""
        if self.tensor_pool is None:
            self.tensor_pool = TensorPool(model_manager.device)

    def _normalize_tensorrt_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        正確歸一化 TensorRT 輸出，維持原版邏輯避免過曝
        TensorRT 的輸出範圍可能不是標準的 [0, 1]，需要先做 min-max 歸一化
        """
        # 對每個樣本分別進行歸一化（批處理版本）
        normalized_tensors = []

        for i in range(tensor.shape[0]):
            sample = tensor[i:i + 1]  # 保持維度 (1, C, H, W)

            # 獲取當前樣本的最小值和最大值
            min_val = torch.min(sample)
            max_val = torch.max(sample)

            # 調試信息：顯示 TensorRT 原始輸出範圍
            print(f"🔍 樣本 {i + 1} TensorRT 原始輸出範圍: [{min_val.item():.4f}, {max_val.item():.4f}]")

            # 避免除零錯誤：只有當 max > min 時才進行歸一化
            if max_val > min_val:
                # min-max 歸一化：將 [min, max] 映射到 [0, 1]
                normalized_sample = (sample - min_val) / (max_val - min_val)
                print(f"✅ 樣本 {i + 1} 歸一化後範圍: [0.0000, 1.0000]")
            else:
                # 如果 min == max，直接設為 0（避免 NaN）
                normalized_sample = torch.zeros_like(sample)
                print(f"⚠️  樣本 {i + 1} min==max，設為零值")

            # 最終 clamp 確保嚴格在 [0, 1] 範圍內
            normalized_sample = torch.clamp(normalized_sample, 0.0, 1.0)
            normalized_tensors.append(normalized_sample)

        # 合併所有歸一化後的樣本
        return torch.cat(normalized_tensors, dim=0)

    def _validate_inputs(self, images: torch.Tensor, masks: torch.Tensor) -> bool:
        """驗證輸入張量"""
        if images.shape[0] != masks.shape[0]:
            print(f"❌ 圖像和遮罩的批次大小不匹配: {images.shape[0]} vs {masks.shape[0]}")
            return False

        if len(images.shape) != 4 or len(masks.shape) != 3:
            print(f"❌ 輸入張量維度不正確: images {images.shape}, masks {masks.shape}")
            return False

        return True

    def _prepare_batch_tensors(self, images: torch.Tensor, masks: torch.Tensor,
                               batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """準備批處理張量"""
        self._initialize_tensor_pool()

        batch_size = len(batch_indices)
        device = model_manager.device

        # 從池中獲取或創建張量
        batch_images = self.tensor_pool.get_tensor(
            (batch_size, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )
        batch_masks = self.tensor_pool.get_tensor(
            (batch_size, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )

        # 填充批處理數據
        for i, idx in enumerate(batch_indices):
            # 處理圖像
            image = images[idx].permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            image_resized = transforms.functional.resize(
                image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
            batch_images[i] = image_resized.squeeze(0).to(device)

            # 處理遮罩
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # 取第一個通道
            mask = mask.unsqueeze(0).unsqueeze(0)  # HW -> BCHW
            mask_resized = transforms.functional.resize(
                mask, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                interpolation=transforms.InterpolationMode.NEAREST
            )
            batch_masks[i] = mask_resized.squeeze(0).to(device)

        return batch_images, batch_masks

    def _postprocess_results(self, results: torch.Tensor, original_shapes: List[Tuple[int, int]],
                             batch_indices: List[int]) -> List[torch.Tensor]:
        """後處理結果張量"""
        processed_results = []

        for i, idx in enumerate(batch_indices):
            result = results[i:i + 1]  # 保持批次維度
            h, w = original_shapes[idx]

            # 縮放回原始尺寸
            result_resized = transforms.functional.resize(
                result, (h, w),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )

            # 轉換為 ComfyUI 格式 (BCHW -> BHWC)
            result_comfy = result_resized.permute(0, 2, 3, 1).cpu()
            processed_results.append(result_comfy)

            # 歸還張量到池中
            if hasattr(self, 'tensor_pool') and self.tensor_pool:
                self.tensor_pool.return_tensor(result.detach())

        return processed_results

    def lama_remover(self, images: torch.Tensor, masks: torch.Tensor,
                     mask_threshold: int, gaussblur_radius: int, invert_mask: bool,
                     batch_size: int = 1, enable_performance_monitor: bool = False):
        """
        優化的核心處理函式
        """
        # 輸入驗證
        if not self._validate_inputs(images, masks):
            return (images,)

        # 性能監控裝飾器
        monitor = performance_monitor if enable_performance_monitor else lambda x: contextmanager(lambda: (yield))()

        with monitor("總處理時間"):
            # 準備處理
            num_images = images.shape[0]
            results = []

            # 記錄原始形狀
            original_shapes = [(images.shape[1], images.shape[2])] * num_images

            # 批處理循環
            for start_idx in range(0, num_images, batch_size):
                end_idx = min(start_idx + batch_size, num_images)
                batch_indices = list(range(start_idx, end_idx))
                current_batch_size = len(batch_indices)

                with monitor(f"批次 {start_idx // batch_size + 1}/{(num_images - 1) // batch_size + 1}"):
                    try:
                        # 準備批處理數據
                        with monitor("數據準備"):
                            batch_images, batch_masks = self._prepare_batch_tensors(
                                images, masks, batch_indices
                            )

                        # 遮罩預處理
                        with monitor("遮罩處理"):
                            if invert_mask:
                                batch_masks = 1.0 - batch_masks

                            # 應用模糊
                            if gaussblur_radius > 0:
                                batch_masks = BlurProcessor.apply_blur(batch_masks, gaussblur_radius)

                            # 二值化
                            threshold = mask_threshold / 255.0
                            batch_masks = (batch_masks > threshold).float()

                        # 模型推理
                        with monitor("TensorRT 推理"):
                            with torch.no_grad():
                                # 調試：顯示輸入數值範圍（保留原版邏輯）
                                print(f"📊 輸入圖片數值範圍: [{batch_images.min().item():.4f}, {batch_images.max().item():.4f}]")
                                print(f"📊 輸入遮罩數值範圍: [{batch_masks.min().item():.4f}, {batch_masks.max().item():.4f}]")

                                batch_results = model_manager.model(batch_images, batch_masks)
                                torch.cuda.synchronize()  # 確保 GPU 操作完成

                        # 結果後處理
                        with monitor("結果後處理"):
                            # 手動歸一化 TensorRT 的輸出（關鍵！維持原版邏輯）
                            # ⚠️ 重要：TensorRT 輸出範圍可能不是 [0, 1]，必須先正確歸一化避免過曝
                            # 原版邏輯：min-max normalization + clamp，不可簡化！
                            batch_results = self._normalize_tensorrt_output(batch_results)

                            # 處理每個結果
                            processed = self._postprocess_results(
                                batch_results, original_shapes, batch_indices
                            )
                            results.extend(processed)

                    except Exception as e:
                        print(f"❌ 批次 {start_idx // batch_size + 1} 處理失敗: {e}")
                        # 備援：返回原始圖像
                        for idx in batch_indices:
                            results.append(images[idx:idx + 1])

                    # 及時清理 GPU 內存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 合併所有結果
        try:
            final_result = torch.cat(results, dim=0)
        except Exception as e:
            print(f"❌ 結果合併失敗: {e}")
            final_result = images

        # 最終內存清理
        if enable_performance_monitor:
            print(f"🧹 最終 GPU 內存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB")

        return (final_result,)


class LamaRemoverIMG(LamaRemover):
    """
    LamaRemover 的 IMAGE 輸入變體，繼承所有優化
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["masks"] = ("IMAGE",)  # 改為 IMAGE 類型
        return base_inputs

    FUNCTION = "lama_remover"


# --- [工具函數] ---
def get_model_info():
    """獲取模型資訊（除錯用）"""
    try:
        info = model_manager.model.get_engine_info()
        return info
    except Exception as e:
        return {"error": str(e)}


def cleanup_resources():
    """清理資源（可選的外部調用）"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 資源清理完成")


# --- ComfyUI 節點註冊 ---
NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "🚀 Big Lama Remover (極限優化版)",
    "LamaRemoverIMG": "🚀 Big Lama Remover IMG (極限優化版)"
}