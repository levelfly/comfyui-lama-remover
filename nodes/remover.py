import torch
from torchvision import transforms
from PIL import ImageFilter
from ..lama import model

try:
    from lama_cpp import _C as custom_cuda_blur
    LAMA_CPP_AVAILABLE = True
    print("✅ 成功載入 custom CUDA blur 模組（極速模式啟用）")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("⚠️ 無法載入 custom CUDA blur 模組，改用 CPU 模糊")

class LamaRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "gaussblur_radius": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        results = []
        images_tensor = images.permute(0, 3, 1, 2)  # NHWC → NCHW

        for i in range(images_tensor.shape[0]):
            try:
                image_tensor = images_tensor[i].unsqueeze(0)
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask_tensor = mask.unsqueeze(0).unsqueeze(0)

                _, _, h, w = image_tensor.shape

                image_512 = transforms.functional.resize(image_tensor, (512, 512), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
                mask_512 = transforms.functional.resize(mask_tensor, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)

                image_gpu = image_512.to(mylama.device)
                mask_gpu = mask_512.to(mylama.device)

                if invert_mask:
                    mask_gpu = 1.0 - mask_gpu

                if gaussblur_radius > 0:
                    if LAMA_CPP_AVAILABLE:
                        mask_gpu = custom_cuda_blur.gaussian_blur(mask_gpu, gaussblur_radius)
                    else:
                        mask_pil = transforms.ToPILImage()(mask_gpu.squeeze().cpu())
                        mask_pil_blur = mask_pil.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
                        mask_gpu = transforms.ToTensor()(mask_pil_blur).unsqueeze(0).to(mylama.device)

                threshold = mask_threshold / 255.0
                final_mask = (mask_gpu > threshold).float()
                coverage_ratio = final_mask.sum() / final_mask.numel()

                print(f"\n🔍 [Image {i}]")
                print(f"Input image range: [{image_gpu.min():.4f}, {image_gpu.max():.4f}]")
                print(f"Input mask range: [{mask_gpu.min():.4f}, {mask_gpu.max():.4f}]")
                print(f"Final mask coverage: {coverage_ratio:.4f}")

                # 🛡️ 小遮罩跳過處理
                if coverage_ratio < 0.01:
                    print(f"⚠️ 遮罩覆蓋率過低，跳過 inpaint，直接返回原圖")
                    result = image_gpu.clone()
                else:
                    result = mylama(image_gpu, final_mask)

                    if torch.isnan(result).any():
                        print("❌ 輸出出現 NaN，返回原圖")
                        result = image_gpu.clone()
                    else:
                        min_val = result.min()
                        max_val = result.max()
                        if max_val > min_val:
                            result = (result - min_val) / (max_val - min_val)
                        result = torch.clamp(result, 0.0, 1.0)

                result_resized = transforms.functional.resize(
                    result, (h, w), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                )
                result_nhwc = result_resized.permute(0, 2, 3, 1).cpu()
                results.append(result_nhwc)

            except Exception as e:
                print(f"❌ 處理第 {i} 張圖時發生錯誤: {e}")
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
