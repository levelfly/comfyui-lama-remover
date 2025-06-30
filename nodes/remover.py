import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter
from ..lama import model  # 假設你這邊的 BigLama 是你封裝的 TRT wrapper

# 嘗試載入自訂 CUDA blur module
try:
    from lama_cpp import _C as custom_cuda_blur
    LAMA_CPP_AVAILABLE = True
    print("✅ Successfully imported custom CUDA blur kernel. Extreme performance mode is ON.")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("⚠️ Custom CUDA blur kernel not found. Falling back to CPU-based blur for stability.")


def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
    mylama = model.BigLama()
    results = []

    # images 的原始格式是 (B, H, W, C)，範圍 [0, 1]，先轉為 PyTorch 慣用的 (B, C, H, W)
    images_tensor = images.permute(0, 3, 1, 2)

    for i in range(images_tensor.shape[0]):
        try:
            image_tensor_single = images_tensor[i].unsqueeze(0)

            current_mask = masks[i]
            if current_mask.ndim == 3:
                mask_tensor_single = current_mask[:, :, 0].unsqueeze(0).unsqueeze(0)
            else:
                mask_tensor_single = current_mask.unsqueeze(0).unsqueeze(0)

            _, _, h, w = image_tensor_single.shape

            # --- 預處理：Resize ---
            image_resized_512 = transforms.functional.resize(
                image_tensor_single, (512, 512),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
            mask_resized_512 = transforms.functional.resize(
                mask_tensor_single, (512, 512),
                interpolation=transforms.InterpolationMode.NEAREST
            )

            image_gpu_512 = image_resized_512.to(mylama.device)
            mask_gpu_512 = mask_resized_512.to(mylama.device)

            if invert_mask:
                mask_gpu_512 = 1.0 - mask_gpu_512

            # --- [極限融合] ---
            if gaussblur_radius > 0:
                if LAMA_CPP_AVAILABLE:
                    mask_gpu_512 = custom_cuda_blur.gaussian_blur(mask_gpu_512, gaussblur_radius)
                else:
                    squeezed_mask_tensor = mask_gpu_512.squeeze()
                    mask_pil = transforms.ToPILImage()(squeezed_mask_tensor.cpu())
                    mask_blurred_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
                    mask_gpu_512 = transforms.ToTensor()(mask_blurred_pil).unsqueeze(0).to(mylama.device)

            threshold = mask_threshold / 255.0
            final_mask_gpu = (mask_gpu_512 > threshold).float()

            # --- [核心修正 1: 輸入正規化] ---
            # 將圖像數據範圍從 [0, 1] 轉換到 [-1, 1]，這是模型需要的
            normalized_image_gpu = image_gpu_512 * 2.0 - 1.0

            # 呼叫 TensorRT 引擎
            # 使用正規化後的圖像作為輸入
            result_gpu_512 = mylama(normalized_image_gpu, final_mask_gpu)

            # --- [核心修正 2: 輸出反正規化] ---
            # 模型輸出的範圍應該是 [-1, 1]，將其還原到 [0, 1]
            denormalized_result_gpu = (result_gpu_512 + 1.0) / 2.0

            # --- [核心修正 3: Clamp] ---
            # 確保數值不會因為浮點數精度問題超過 [0, 1] 範圍
            denormalized_result_gpu = torch.clamp(denormalized_result_gpu, 0.0, 1.0)

            # 將結果 resize 回原始尺寸
            # 注意：這裡要用我們修正後的 denormalized_result_gpu
            result_resized_gpu = transforms.functional.resize(
                denormalized_result_gpu, (h, w),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )

            # 轉換回 ComfyUI 的 (B, H, W, C) 格式並移至 CPU
            result_comfy = result_resized_gpu.permute(0, 2, 3, 1).cpu()
            results.append(result_comfy)

        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")
            results.append(images[i:i + 1])

    return (torch.cat(results, dim=0),)


class LamaRemoverIMG(LamaRemover):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("IMAGE",),
                "mask_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "gaussblur_radius": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "lama_remover"


NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover",
    "LamaRemoverIMG": "Big lama Remover(IMG)"
}
