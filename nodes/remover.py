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

    def normalize_input(self, tensor):
        """確保輸入張量在正確的範圍內 [0, 1]"""
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.float()  # 確保是 float32

    def denormalize_output(self, tensor):
        """將輸出張量轉換到正確範圍並處理精度問題"""
        # 處理 FP16/TensorRT 可能的數值異常
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

        # 確保數值在合理範圍內
        tensor = torch.clamp(tensor, 0, 1)

        return tensor

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            print(f"input image size: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            # 確保輸入圖像在正確範圍內
            pt_image = self.normalize_input(pt_image)
            print(f"Normalized image range: [{pt_image.min().item():.4f}, {pt_image.max().item():.4f}]")

            mask = mask.unsqueeze(0)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"input mask size: {ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # gaussian Blur
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)
            # 確保遮罩在正確範圍內
            pt_mask = self.normalize_input(pt_mask)
            print(f"Normalized mask range: [{pt_mask.min().item():.4f}, {pt_mask.max().item():.4f}]")

            # lama 模型推理
            result = mylama(pt_image, pt_mask)

            # 處理 TensorRT 輸出的維度和數值問題
            print(f"Raw result shape: {result.shape}")
            print(f"Raw result range: [{result.min().item():.4f}, {result.max().item():.4f}]")
            print(f"Raw result dtype: {result.dtype}")

            # 移除 batch 維度
            if result.dim() == 4:
                result = result.squeeze(0)
            elif result.dim() == 3 and result.shape[0] == 1:
                result = result.squeeze(0)

            # 轉換到 CPU 並確保是 float32
            result = result.cpu().float()

            # 處理 FP16 轉換可能的數值問題
            result = self.denormalize_output(result)

            print(f"Processed result shape: {result.shape}")
            print(f"Processed result range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # 轉換為 PIL 圖像
            img_result = ten2pil(result)

            # crop into the original size
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # turn to comfyui tensor
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

    def normalize_input(self, tensor):
        """確保輸入張量在正確的範圍內 [0, 1]"""
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.float()  # 確保是 float32

    def denormalize_output(self, tensor):
        """將輸出張量轉換到正確範圍並處理精度問題"""
        # 處理 FP16/TensorRT 可能的數值異常
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

        # 確保數值在合理範圍內
        tensor = torch.clamp(tensor, 0, 1)

        return tensor

    def lama_remover_IMG(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            ori_image = tensor2pil(image)
            print(f"input image size: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            # 確保輸入圖像在正確範圍內
            pt_image = self.normalize_input(pt_image)
            print(f"Normalized image range: [{pt_image.min().item():.4f}, {pt_image.max().item():.4f}]")

            mask = mask.movedim(0, -1).movedim(0, -1)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"input mask size: {ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("resize mask")
                p_mask = p_mask.resize(p_image.size)

            # invert mask
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # gaussian Blur
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # mask_threshold
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)
            # 確保遮罩在正確範圍內
            pt_mask = self.normalize_input(pt_mask)
            print(f"Normalized mask range: [{pt_mask.min().item():.4f}, {pt_mask.max().item():.4f}]")

            # lama 模型推理
            result = mylama(pt_image, pt_mask)

            # 處理 TensorRT 輸出的維度和數值問題
            print(f"Raw result shape: {result.shape}")
            print(f"Raw result range: [{result.min().item():.4f}, {result.max().item():.4f}]")
            print(f"Raw result dtype: {result.dtype}")

            # 移除 batch 維度
            if result.dim() == 4:
                result = result.squeeze(0)
            elif result.dim() == 3 and result.shape[0] == 1:
                result = result.squeeze(0)

            # 轉換到 CPU 並確保是 float32
            result = result.cpu().float()

            # 處理 FP16 轉換可能的數值問題
            result = self.denormalize_output(result)

            print(f"Processed result shape: {result.shape}")
            print(f"Processed result range: [{result.min().item():.4f}, {result.max().item():.4f}]")

            # 轉換為 PIL 圖像
            img_result = ten2pil(result)

            # crop into the original size
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # turn to comfyui tensor
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