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


# --- [圖像變換記錄] ---
class ImageTransformInfo:
    """記錄圖像變換信息，用於精確的填充與裁切"""

    def __init__(self, original_h: int, original_w: int,
                 scaled_h: int, scaled_w: int,
                 x_offset: int, y_offset: int):
        self.original_h = original_h
        self.original_w = original_w
        self.scaled_h = scaled_h
        self.scaled_w = scaled_w
        self.x_offset = x_offset
        self.y_offset = y_offset


# --- [高品質圖像處理器] ---
class HighQualityImageProcessor:
    """
    專業級圖像處理器，實現保持長寬比的填充與裁切
    """

    @staticmethod
    def get_interpolation_mode(mode_str: str):
        """獲取插值模式"""
        if mode_str == "BICUBIC":
            return transforms.InterpolationMode.BICUBIC
        else:
            return transforms.InterpolationMode.BILINEAR

    @staticmethod
    def pad_to_square(image: torch.Tensor, target_size: int = MODEL_INPUT_SIZE,
                      interpolation_mode=transforms.InterpolationMode.BICUBIC) -> Tuple[torch.Tensor, ImageTransformInfo]:
        """
        將圖像填充為正方形，保持長寬比
        返回：(填充後的圖像, 變換信息)
        """
        # 輸入: (C, H, W)
        _, original_h, original_w = image.shape

        # 計算縮放比例，使最長邊等於target_size
        scale = target_size / max(original_h, original_w)
        scaled_h = int(original_h * scale)
        scaled_w = int(original_w * scale)

        # 縮放圖像，保持長寬比
        image_scaled = transforms.functional.resize(
            image.unsqueeze(0), (scaled_h, scaled_w),
            interpolation=interpolation_mode,
            antialias=True
        ).squeeze(0)

        # 計算填充位置（居中）
        x_offset = (target_size - scaled_w) // 2
        y_offset = (target_size - scaled_h) // 2

        # 創建黑色背景並填充
        padded_image = torch.zeros(image.shape[0], target_size, target_size,
                                   dtype=image.dtype, device=image.device)
        padded_image[:, y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = image_scaled

        # 記錄變換信息
        transform_info = ImageTransformInfo(
            original_h, original_w, scaled_h, scaled_w, x_offset, y_offset
        )

        return padded_image, transform_info

    @staticmethod
    def crop_and_restore(result: torch.Tensor, transform_info: ImageTransformInfo,
                         interpolation_mode=transforms.InterpolationMode.BICUBIC) -> torch.Tensor:
        """
        從結果中裁切出有效區域並恢復到原始尺寸
        """
        # 輸入: (C, H, W) 或 (1, C, H, W)
        if result.dim() == 4:
            result = result.squeeze(0)  # 移除batch維度

        # 裁切出有效區域
        y_start = transform_info.y_offset
        y_end = transform_info.y_offset + transform_info.scaled_h
        x_start = transform_info.x_offset
        x_end = transform_info.x_offset + transform_info.scaled_w

        cropped = result[:, y_start:y_end, x_start:x_end]

        # 縮放回原始尺寸
        restored = transforms.functional.resize(
            cropped.unsqueeze(0),
            (transform_info.original_h, transform_info.original_w),
            interpolation=interpolation_mode,
            antialias=True
        ).squeeze(0)

        return restored


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
                "aggressive_normalization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "啟用激進正規化（僅在輸出過暗/過曝時使用）"
                }),
                "interpolation_mode": (["BICUBIC", "BILINEAR"], {
                    "default": "BICUBIC",
                    "tooltip": "插值演算法：BICUBIC品質更佳，BILINEAR速度更快"
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

    def _validate_inputs(self, images: torch.Tensor, masks: torch.Tensor, is_image_mask: bool = False) -> bool:
        """驗證輸入張量"""
        if images.shape[0] != masks.shape[0]:
            print(f"❌ 圖像和遮罩的批次大小不匹配: {images.shape[0]} vs {masks.shape[0]}")
            return False

        # 檢查維度
        if len(images.shape) != 4:
            print(f"❌ 圖像張量維度不正確: {images.shape}，期望 4 維 (B,H,W,C)")
            return False

        if is_image_mask:
            # IMAGE 類型遮罩應該是 4 維 (B,H,W,C)
            if len(masks.shape) != 4:
                print(f"❌ IMAGE 類型遮罩張量維度不正確: {masks.shape}，期望 4 維 (B,H,W,C)")
                return False
        else:
            # MASK 類型應該是 3 維 (B,H,W)
            if len(masks.shape) != 3:
                print(f"❌ MASK 類型遮罩張量維度不正確: {masks.shape}，期望 3 維 (B,H,W)")
                return False

        return True

    def _prepare_batch_tensors(self, images: torch.Tensor, masks: torch.Tensor,
                               batch_indices: List[int], is_image_mask: bool = False,
                               interpolation_mode_str: str = "BICUBIC") -> Tuple[torch.Tensor, torch.Tensor, List[ImageTransformInfo]]:
        """
        準備批處理張量 - 使用保持長寬比的填充策略
        返回：(批處理圖像, 批處理遮罩, 變換信息列表)
        """
        self._initialize_tensor_pool()

        batch_size = len(batch_indices)
        device = model_manager.device
        interpolation_mode = HighQualityImageProcessor.get_interpolation_mode(interpolation_mode_str)

        # 從池中獲取或創建張量
        batch_images = self.tensor_pool.get_tensor(
            (batch_size, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )
        batch_masks = self.tensor_pool.get_tensor(
            (batch_size, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )

        transform_infos = []

        # 填充批處理數據
        for i, idx in enumerate(batch_indices):
            # 處理圖像 - 保持長寬比的填充
            image = images[idx].permute(2, 0, 1)  # HWC -> CHW

            # 【關鍵改進】使用保持長寬比的填充
            padded_image, transform_info = HighQualityImageProcessor.pad_to_square(
                image, MODEL_INPUT_SIZE, interpolation_mode
            )

            batch_images[i] = padded_image.to(device)
            transform_infos.append(transform_info)

            print(f"🖼️  圖像 {idx}: {transform_info.original_h}x{transform_info.original_w} → "
                  f"{transform_info.scaled_h}x{transform_info.scaled_w} (填充到 {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})")

            # 處理遮罩 - 同樣使用保持長寬比的邏輯
            mask = masks[idx]

            if is_image_mask:
                # IMAGE 類型遮罩 (H, W, C) - 需要轉換為灰階
                if mask.ndim == 3 and mask.shape[2] > 1:
                    if mask.shape[2] >= 3:
                        mask = 0.299 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.114 * mask[:, :, 2]
                    else:
                        mask = mask[:, :, 0]
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                print(f"🎭 IMAGE 類型遮罩處理完成，形狀: {mask.shape}")
            else:
                # MASK 類型遮罩 (H, W)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                print(f"🎭 MASK 類型遮罩處理完成，形狀: {mask.shape}")

            # 對遮罩應用相同的填充邏輯
            mask_chw = mask.unsqueeze(0)  # HW -> CHW
            padded_mask, _ = HighQualityImageProcessor.pad_to_square(
                mask_chw, MODEL_INPUT_SIZE, transforms.InterpolationMode.NEAREST  # 遮罩用最近鄰
            )

            batch_masks[i] = padded_mask.to(device)

            # 調試信息
            print(f"📊 樣本 {idx}: 遮罩值範圍 [{padded_mask.min().item():.3f}, {padded_mask.max().item():.3f}]")

        return batch_images, batch_masks, transform_infos

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
                     batch_size: int = 1, enable_performance_monitor: bool = False,
                     aggressive_normalization: bool = False, interpolation_mode: str = "BICUBIC",
                     is_image_mask: bool = False):
        """
        【專業級品質版本】極限優化的核心處理函式
        新增品質改進：
        - 保持長寬比的填充與裁切
        - 可選的正規化策略
        - BICUBIC高品質插值
        """
        # 輸入驗證
        if not self._validate_inputs(images, masks, is_image_mask):
            return (images,)

        # 性能監控裝飾器
        monitor = performance_monitor if enable_performance_monitor else lambda x: contextmanager(lambda: (yield))()

        with monitor("總處理時間"):
            # 準備處理
            num_images = images.shape[0]
            results = []

            print(f"🎯 處理模式: {'IMAGE 類型遮罩' if is_image_mask else 'MASK 類型遮罩'}")
            print(f"🎨 插值模式: {interpolation_mode}")
            print(f"🔧 正規化策略: {'激進拉伸' if aggressive_normalization else '溫和鉗位'}")

            # 批處理循環
            for start_idx in range(0, num_images, batch_size):
                end_idx = min(start_idx + batch_size, num_images)
                batch_indices = list(range(start_idx, end_idx))
                current_batch_size = len(batch_indices)

                with monitor(f"批次 {start_idx // batch_size + 1}/{(num_images - 1) // batch_size + 1}"):
                    try:
                        # 【品質改進】準備批處理數據 - 使用保持長寬比的填充
                        with monitor("高品質數據準備"):
                            batch_images, batch_masks, transform_infos = self._prepare_batch_tensors(
                                images, masks, batch_indices, is_image_mask, interpolation_mode
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
                                # 調試：顯示輸入數值範圍
                                print(f"📊 輸入圖片數值範圍: [{batch_images.min().item():.4f}, {batch_images.max().item():.4f}]")
                                print(f"📊 輸入遮罩數值範圍: [{batch_masks.min().item():.4f}, {batch_masks.max().item():.4f}]")

                                batch_results = model_manager.model(batch_images, batch_masks)
                                torch.cuda.synchronize()

                        # 【品質改進】結果後處理
                        with monitor("高品質結果後處理"):
                            # 【選擇性正規化】根據用戶設定選擇策略
                            batch_results = self._normalize_tensorrt_output(
                                batch_results, aggressive_normalization
                            )

                            # 【精確恢復】使用裁切和縮放恢復原始尺寸
                            processed = self._postprocess_results(
                                batch_results, transform_infos, batch_indices, interpolation_mode
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
    專門處理 IMAGE 類型的遮罩輸入
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["masks"] = ("IMAGE",)  # 改為 IMAGE 類型
        return base_inputs

    FUNCTION = "lama_remover_img"

    def lama_remover_img(self, images: torch.Tensor, masks: torch.Tensor,
                         mask_threshold: int, gaussblur_radius: int, invert_mask: bool,
                         batch_size: int = 1, enable_performance_monitor: bool = False):
        """
        IMAGE 類型遮罩的處理入口，調用父類方法並標記遮罩類型
        """
        return self.lama_remover(
            images=images,
            masks=masks,
            mask_threshold=mask_threshold,
            gaussblur_radius=gaussblur_radius,
            invert_mask=invert_mask,
            batch_size=batch_size,
            enable_performance_monitor=enable_performance_monitor,
            is_image_mask=True  # 關鍵：標記為 IMAGE 類型遮罩
        )


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