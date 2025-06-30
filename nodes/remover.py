# /nodes/remover.py (修復版本)

import torch
import torch.nn.functional as F
from torchvision import transforms
from ..lama import model

# 修復匯入語法錯誤
try:
    from lama_cpp import C as custom_cuda_blur

    LAMA_CPP_AVAILABLE = True
    print("Successfully imported custom CUDA blur kernel. Extreme performance mode is ON.")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("Custom CUDA blur kernel not found. Falling back to CPU-based blur for stability.")
    from PIL import ImageFilter


class LamaRemover:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gaussblur_radius": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        results = []

        # 確保輸入圖像在正確範圍 [0, 1]
        images_tensor = images.permute(0, 3, 1, 2)

        # 調試信息
        print(f"Input images shape: {images_tensor.shape}")
        print(f"Input images range: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]")
        print(f"Input masks shape: {masks.shape}")
        print(f"Input masks range: [{masks.min():.3f}, {masks.max():.3f}]")

        for i in range(images_tensor.shape[0]):
            try:
                image_tensor_single = images_tensor[i].unsqueeze(0)

                # 處理遮罩
                current_mask = masks[i]
                if current_mask.ndim == 3:
                    mask_tensor_single = current_mask[:, :, 0].unsqueeze(0).unsqueeze(0)
                else:
                    mask_tensor_single = current_mask.unsqueeze(0).unsqueeze(0)

                _, _, h, w = image_tensor_single.shape

                # Resize到512x512
                image_resized_512 = transforms.functional.resize(
                    image_tensor_single, (512, 512),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
                mask_resized_512 = transforms.functional.resize(
                    mask_tensor_single, (512, 512),
                    interpolation=transforms.InterpolationMode.NEAREST
                )

                # 移動到GPU
                image_gpu_512 = image_resized_512.to(mylama.device)
                mask_gpu_512 = mask_resized_512.to(mylama.device)

                # 確保圖像在 [0, 1] 範圍
                image_gpu_512 = torch.clamp(image_gpu_512, 0.0, 1.0)

                # 確保遮罩在 [0, 1] 範圍
                mask_gpu_512 = torch.clamp(mask_gpu_512, 0.0, 1.0)

                if invert_mask:
                    mask_gpu_512 = 1.0 - mask_gpu_512

                # 高斯模糊處理
                if gaussblur_radius > 0:
                    if LAMA_CPP_AVAILABLE:
                        mask_gpu_512 = custom_cuda_blur.gaussian_blur(mask_gpu_512, gaussblur_radius)
                    else:
                        # CPU備用方案
                        squeezed_mask_tensor = mask_gpu_512.squeeze()
                        mask_pil = transforms.ToPILImage()(squeezed_mask_tensor.cpu())
                        mask_blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
                        mask_gpu_512 = transforms.ToTensor()(mask_blurred_pil).unsqueeze(0).to(mylama.device)

                # 修復閾值處理 - 使用更合理的邏輯
                final_mask_gpu = (mask_gpu_512 > mask_threshold).float()

                # 調試信息
                print(f"Final mask range: [{final_mask_gpu.min():.3f}, {final_mask_gpu.max():.3f}]")
                print(f"Mask coverage: {final_mask_gpu.mean():.3f}")

                # 確保LaMa模型輸入格式正確
                # LaMa通常期望圖像在 [0, 1] 範圍，遮罩為二進制
                result_gpu_512 = mylama(image_gpu_512, final_mask_gpu)

                # 修復正規化邏輯
                # 檢查輸出是否已經在合理範圍內
                print(f"Model output range before normalization: [{result_gpu_512.min():.3f}, {result_gpu_512.max():.3f}]")

                # 只有當輸出超出 [0, 1] 範圍時才進行正規化
                if result_gpu_512.min() < 0 or result_gpu_512.max() > 1:
                    # 如果模型輸出在 [-1, 1] 範圍，轉換到 [0, 1]
                    if result_gpu_512.min() >= -1 and result_gpu_512.max() <= 1:
                        result_gpu_512 = (result_gpu_512 + 1.0) / 2.0
                    else:
                        # 通用正規化
                        min_val = torch.min(result_gpu_512)
                        max_val = torch.max(result_gpu_512)
                        if max_val > min_val:
                            result_gpu_512 = (result_gpu_512 - min_val) / (max_val - min_val)

                # 確保最終輸出在 [0, 1] 範圍
                result_gpu_512 = torch.clamp(result_gpu_512, 0.0, 1.0)

                print(f"Final output range: [{result_gpu_512.min():.3f}, {result_gpu_512.max():.3f}]")

                # Resize回原始尺寸
                result_resized_gpu = transforms.functional.resize(
                    result_gpu_512, (h, w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )

                # 轉換回ComfyUI格式
                result_comfy = result_resized_gpu.permute(0, 2, 3, 1).cpu()
                results.append(result_comfy)

            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                # 返回原始圖像作為備用
                results.append(images[i:i + 1])

        final_result = torch.cat(results, dim=0)
        print(f"Final result shape: {final_result.shape}")
        print(f"Final result range: [{final_result.min():.3f}, {final_result.max():.3f}]")

        return (final_result,)


class LamaRemoverIMG(LamaRemover):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("IMAGE",),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gaussblur_radius": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "lama_remover"


NODE_CLASS_MAPPINGS = {"LamaRemover": LamaRemover, "LamaRemoverIMG": LamaRemoverIMG}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover",
    "LamaRemoverIMG": "Big lama Remover(IMG)"
}