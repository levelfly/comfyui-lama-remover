# /nodes/remover.py (Final Corrected Version)

import torch
import torch.nn.functional as F
from torchvision import transforms
from ..lama import model

try:
    from lama_cpp import _C as custom_cuda_blur

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
                "mask_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
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

                image_resized_512 = transforms.functional.resize(
                    image_tensor_single, (512, 512),
                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                )
                mask_resized_512 = transforms.functional.resize(
                    mask_tensor_single, (512, 512),
                    interpolation=transforms.InterpolationMode.NEAREST
                )

                image_gpu_512 = image_resized_512.to(mylama.device)
                mask_gpu_512 = mask_resized_512.to(mylama.device)

                if invert_mask:
                    mask_gpu_512 = 1.0 - mask_gpu_512

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

                # 將輸入影像從 [0, 1] 範圍轉換到 [-1, 1] (根據日誌，這一步是正確的)
                image_gpu_512_norm = image_gpu_512 * 2.0 - 1.0

                # 呼叫 TensorRT 引擎
                result_gpu_512 = mylama(image_gpu_512_norm, final_mask_gpu)

                ### FINAL DEBUG ###
                # 根據日誌，模型輸出範圍是 [0, 255]。我們需要將其轉換回 ComfyUI 需要的 [0, 1] 範圍。
                # 正確的操作是直接除以 255。
                result_gpu_512 = result_gpu_512 / 255.0

                # 使用 clamp 確保數值穩定在 [0, 1] 範圍內，防止微小的浮點數誤差
                result_gpu_512 = torch.clamp(result_gpu_512, 0.0, 1.0)

                # 將結果 resize 回原始尺寸
                result_resized_gpu = transforms.functional.resize(
                    result_gpu_512, (h, w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )

                result_comfy = result_resized_gpu.permute(0, 2, 3, 1).cpu()
                results.append(result_comfy)

            except Exception as e:
                print(f"Error processing image {i + 1}: {e}")
                results.append(images[i:i + 1])

        return (torch.cat(results, dim=0),)


class LamaRemoverIMG(LamaRemover):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), "masks": ("IMAGE",), "mask_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                             "gaussblur_radius": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}), "invert_mask": ("BOOLEAN", {"default": False}), }}

    FUNCTION = "lama_remover"


NODE_CLASS_MAPPINGS = {"LamaRemover": LamaRemover, "LamaRemoverIMG": LamaRemoverIMG}
NODE_DISPLAY_NAME_MAPPINGS = {"LamaRemover": "Big lama Remover", "LamaRemoverIMG": "Big lama Remover(IMG)"}