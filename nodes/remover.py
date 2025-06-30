# remover.py (修改後)

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

            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # --- [開始] LaMa TRT 模型歸一化修改 ---

            # 1. 輸入歸一化: 將圖像張量從 [0, 1] 範圍轉換到 [-1, 1] 範圍
            #    這是 LaMa 模型訓練時使用的標準。
            print(f"歸一化前，圖片張量範圍: [{pt_image.min():.4f}, {pt_image.max():.4f}]")
            pt_image = pt_image * 2.0 - 1.0
            print(f"歸一化後，圖片張量範圍: [{pt_image.min():.4f}, {pt_image.max():.4f}]")

            # 注意: 遮罩 (pt_mask) 通常保持在 [0, 1] 範圍即可，無需修改。

            # lama
            result = mylama(pt_image, pt_mask)

            # 2. 輸出反歸一化: 將模型輸出的 [-1, 1] 範圍轉換回 [0, 1] 範圍
            print(f"反歸一化前，輸出張量範圍: [{result.min():.4f}, {result.max():.4f}]")
            result = (result + 1.0) / 2.0

            # 3. 裁剪值: 確保數值在 [0, 1] 範圍內，防止浮點數精度問題
            result = torch.clamp(result, 0, 1)
            print(f"反歸一化後，輸出張量範圍: [{result.min():.4f}, {result.max():.4f}]")

            # --- [結束] LaMa TRT 模型歸一化修改 ---

            img_result = ten2pil(result)

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

            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)

            # --- [開始] LaMa TRT 模型歸一化修改 ---

            # 1. 輸入歸一化: 將圖像張量從 [0, 1] 範圍轉換到 [-1, 1] 範圍
            print(f"歸一化前，圖片張量範圍: [{pt_image.min():.4f}, {pt_image.max():.4f}]")
            pt_image = pt_image * 2.0 - 1.0
            print(f"歸一化後，圖片張量範圍: [{pt_image.min():.4f}, {pt_image.max():.4f}]")

            # lama
            result = mylama(pt_image, pt_mask)

            # 2. 輸出反歸一化: 將模型輸出的 [-1, 1] 範圍轉換回 [0, 1] 範圍
            print(f"反歸一化前，輸出張量範圍: [{result.min():.4f}, {result.max():.4f}]")
            result = (result + 1.0) / 2.0

            # 3. 裁剪值: 確保數值在 [0, 1] 範圍內
            result = torch.clamp(result, 0, 1)
            print(f"反歸一化後，輸出張量範圍: [{result.min():.4f}, {result.max():.4f}]")

            # --- [結束] LaMa TRT 模型歸一化修改 ---

            img_result = ten2pil(result)

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