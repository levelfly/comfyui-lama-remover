# remover.py (已修復 ValueError)

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
            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.unsqueeze(0)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                p_mask = p_mask.resize(p_image.size)

            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            # --- LaMa TRT 模型歸一化 ---
            # 輸入歸一化: 將 [0, 1] 轉換到 [-1, 1]
            pt_image = pt_image * 2.0 - 1.0

            # Lama 模型推理
            result = mylama(pt_image, pt_mask)

            # 輸出反歸一化: 將 [-1, 1] 轉換回 [0, 1]
            result = (result + 1.0) / 2.0
            result = torch.clamp(result, 0, 1)

            # --- [開始] ValueError 修復 ---
            # 模型輸出的 result 是 4D Tensor (N, C, H, W)，N=1
            # ToPILImage() 需要 3D Tensor (C, H, W)
            # 因此我們從批次中選取第 0 個元素
            img_result = ten2pil(result[0])
            # --- [結束] ValueError 修復 ---

            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

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
            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.movedim(0, -1).movedim(0, -1)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                p_mask = p_mask.resize(p_image.size)

            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            # --- LaMa TRT 模型歸一化 ---
            # 輸入歸一化: 將 [0, 1] 轉換到 [-1, 1]
            pt_image = pt_image * 2.0 - 1.0

            # Lama 模型推理
            result = mylama(pt_image, pt_mask)

            # 輸出反歸一化: 將 [-1, 1] 轉換回 [0, 1]
            result = (result + 1.0) / 2.0
            result = torch.clamp(result, 0, 1)

            # --- [開始] ValueError 修復 ---
            # 模型輸出的 result 是 4D Tensor (N, C, H, W)，N=1
            # ToPILImage() 需要 3D Tensor (C, H, W)
            # 因此我們從批次中選取第 0 個元素
            img_result = ten2pil(result[0])
            # --- [結束] ValueError 修復 ---

            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

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