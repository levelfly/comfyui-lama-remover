from PIL import ImageOps, ImageFilter
import torch
from ..utils import cropimage, padimage, padmask, tensor2pil, pil2tensor, cropimage, pil2comfy
from ..lama import model
from torchvision import transforms


class LamaRemover:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            print(f"input image size :{ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.unsqueeze(0)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"input mask size :{ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            # 反转遮罩
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # gaussian Blur
            # 高斯模糊遮罩（模糊的是白色）
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            # 遮罩阈值，越大越强
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # lama
            # lama模型
            result = mylama(pt_image, pt_mask)

            print(f"Result tensor shape: {result.shape}")
            print(f"Result value range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # TensorRT 輸出的手動歸一化處理（參考提供的代碼）
            min_val = torch.min(result)
            max_val = torch.max(result)
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)
            result = torch.clamp(result, 0.0, 1.0)

            print(f"After normalization value range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # 如果輸出尺寸與預期不符，調整回填充後的尺寸
            if result.shape[-2:] != (p_image.size[1], p_image.size[0]):  # (height, width)
                from torchvision import transforms
                target_h, target_w = p_image.size[1], p_image.size[0]
                print(f"Resizing result from {result.shape[-2:]} to ({target_h}, {target_w})")
                result = transforms.functional.resize(
                    result, (target_h, target_w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )

            # 處理維度：確保為 3 維 (C, H, W) 用於 ToPILImage
            if result.dim() == 4:
                # 如果是 4 維 (batch, channel, height, width)，取第一個 batch
                result = result[0]  # 或者使用 result.squeeze(0)

            print(f"Final result tensor shape: {result.shape}")

            img_result = ten2pil(result)

            # crop into the original size
            # 裁剪成输入大小
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # turn to comfyui tensor
            # 变成comfyui格式（i,h,w,c）
            i = pil2comfy(img_result)
            results.append(i)

        return (torch.cat(results, dim=0),)


class LamaRemoverIMG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("IMAGE",),
                "mask_threshold": ("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover_IMG"

    def lama_remover_IMG(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            print(f"input image size :{ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.movedim(0, -1).movedim(0, -1)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"input mask size :{ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            # 反转遮罩
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # gaussian Blur
            # 高斯模糊遮罩（模糊的是黑色所以需要反转操作）
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            # 遮罩阈值，越大越强
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # lama
            # lama模型
            result = mylama(pt_image, pt_mask)

            print(f"Result tensor shape: {result.shape}")
            print(f"Result value range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # TensorRT 輸出的手動歸一化處理（參考提供的代碼）
            min_val = torch.min(result)
            max_val = torch.max(result)
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)
            result = torch.clamp(result, 0.0, 1.0)

            print(f"After normalization value range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # 如果輸出尺寸與預期不符，調整回填充後的尺寸
            if result.shape[-2:] != (p_image.size[1], p_image.size[0]):  # (height, width)
                from torchvision import transforms
                target_h, target_w = p_image.size[1], p_image.size[0]
                print(f"Resizing result from {result.shape[-2:]} to ({target_h}, {target_w})")
                result = transforms.functional.resize(
                    result, (target_h, target_w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )

            # 處理維度：確保為 3 維 (C, H, W) 用於 ToPILImage
            if result.dim() == 4:
                # 如果是 4 維 (batch, channel, height, width)，取第一個 batch
                result = result[0]  # 或者使用 result.squeeze(0)

            print(f"Final result tensor shape: {result.shape}")

            img_result = ten2pil(result)

            # crop into the original size
            # 裁剪成输入大小
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # turn to comfyui tensor
            # 变成comfyui格式（i,h,w,c）
            i = pil2comfy(img_result)
            results.append(i)

        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover",
    "LamaRemoverIMG": "Big lama Remover(IMG)"
}