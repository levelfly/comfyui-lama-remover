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
                # 添加歸一化選項進行測試
                "input_normalization": (["none", "0_1", "imagenet", "0_255"], {"default": "none"}),
                "output_denormalization": (["none", "0_1", "imagenet", "0_255"], {"default": "none"}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover"

    def apply_input_normalization(self, tensor, norm_type):
        """應用不同的輸入歸一化方式"""
        print(f"輸入歸一化前範圍: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")

        if norm_type == "none":
            # 不做任何歸一化
            result = tensor
        elif norm_type == "0_1":
            # 確保在 [0,1] 範圍內
            if tensor.max() > 1.0:
                result = tensor / 255.0
            else:
                result = tensor
        elif norm_type == "imagenet":
            # ImageNet 標準化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # 先確保在 [0,1] 範圍內
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            result = (tensor - mean) / std
        elif norm_type == "0_255":
            # 確保在 [0,255] 範圍內
            if tensor.max() <= 1.0:
                result = tensor * 255.0
            else:
                result = tensor

        print(f"輸入歸一化後範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
        return result

    def apply_output_denormalization(self, tensor, norm_type):
        """應用不同的輸出反歸一化方式"""
        print(f"輸出反歸一化前範圍: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")

        if norm_type == "none":
            # 不做任何處理，但確保在合理範圍內
            result = torch.clamp(tensor, 0, 1)
        elif norm_type == "0_1":
            # 確保輸出在 [0,1] 範圍內
            result = torch.clamp(tensor, 0, 1)
        elif norm_type == "imagenet":
            # ImageNet 反標準化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            result = tensor * std + mean
            result = torch.clamp(result, 0, 1)
        elif norm_type == "0_255":
            # 從 [0,255] 轉換到 [0,1]
            result = torch.clamp(tensor, 0, 255) / 255.0

        print(f"輸出反歸一化後範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
        return result

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask, input_normalization, output_denormalization):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            print(f"\n=== 處理新圖像 ===")

            # 原始圖像處理
            ori_image = tensor2pil(image)
            print(f"原始圖像尺寸: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            print(f"填充後圖像張量形狀: {pt_image.shape}")
            print(f"填充後圖像原始範圍: [{pt_image.min().item():.6f}, {pt_image.max().item():.6f}]")

            # 應用輸入歸一化
            pt_image = self.apply_input_normalization(pt_image, input_normalization)

            # 遮罩處理
            mask = mask.unsqueeze(0)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"原始遮罩尺寸: {ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("調整遮罩尺寸")
                p_mask = p_mask.resize(p_image.size)

            # 反轉遮罩
            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            # 高斯模糊
            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            # 遮罩閾值
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)
            print(f"處理後遮罩張量形狀: {pt_mask.shape}")
            print(f"處理後遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            # 遮罩歸一化（通常遮罩範圍是 [0,1]）
            if pt_mask.max() > 1.0:
                pt_mask = pt_mask / 255.0

            print(f"歸一化後遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            # 確保張量類型和設備一致
            pt_image = pt_image.float()
            pt_mask = pt_mask.float()

            print(f"輸入模型前 - 圖像: {pt_image.shape}, 遮罩: {pt_mask.shape}")
            print(f"輸入模型前 - 圖像範圍: [{pt_image.min().item():.6f}, {pt_image.max().item():.6f}]")
            print(f"輸入模型前 - 遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            # LaMa 模型推理
            result = mylama(pt_image, pt_mask)

            print(f"模型輸出形狀: {result.shape}")
            print(f"模型輸出原始範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
            print(f"模型輸出數據類型: {result.dtype}")

            # 處理維度
            if result.dim() == 4:
                result = result.squeeze(0)
            elif result.dim() == 3 and result.shape[0] == 1:
                result = result.squeeze(0)

            # 轉換到 CPU
            result = result.cpu().float()

            # 處理可能的異常值
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

            print(f"清理後輸出範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")

            # 應用輸出反歸一化
            result = self.apply_output_denormalization(result, output_denormalization)

            # 轉換為 PIL 圖像
            try:
                img_result = ten2pil(result)
                print(f"成功轉換為 PIL 圖像，尺寸: {img_result.size}")
            except Exception as e:
                print(f"轉換 PIL 圖像時發生錯誤: {e}")
                print(f"結果張量形狀: {result.shape}")
                print(f"結果張量範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
                # 緊急處理：強制歸一化到 [0,1]
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)
                img_result = ten2pil(result)

            # 裁剪到原始尺寸
            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            # 轉換為 ComfyUI 格式
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
                # 添加歸一化選項
                "input_normalization": (["none", "0_1", "imagenet", "0_255"], {"default": "none"}),
                "output_denormalization": (["none", "0_1", "imagenet", "0_255"], {"default": "none"}),
            },
        }

    CATEGORY = "LamaRemover"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "lama_remover_IMG"

    def apply_input_normalization(self, tensor, norm_type):
        """應用不同的輸入歸一化方式"""
        print(f"輸入歸一化前範圍: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")

        if norm_type == "none":
            result = tensor
        elif norm_type == "0_1":
            if tensor.max() > 1.0:
                result = tensor / 255.0
            else:
                result = tensor
        elif norm_type == "imagenet":
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            result = (tensor - mean) / std
        elif norm_type == "0_255":
            if tensor.max() <= 1.0:
                result = tensor * 255.0
            else:
                result = tensor

        print(f"輸入歸一化後範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
        return result

    def apply_output_denormalization(self, tensor, norm_type):
        """應用不同的輸出反歸一化方式"""
        print(f"輸出反歸一化前範圍: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")

        if norm_type == "none":
            result = torch.clamp(tensor, 0, 1)
        elif norm_type == "0_1":
            result = torch.clamp(tensor, 0, 1)
        elif norm_type == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            result = tensor * std + mean
            result = torch.clamp(result, 0, 1)
        elif norm_type == "0_255":
            result = torch.clamp(tensor, 0, 255) / 255.0

        print(f"輸出反歸一化後範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
        return result

    def lama_remover_IMG(self, images, masks, mask_threshold, gaussblur_radius, invert_mask, input_normalization, output_denormalization):
        mylama = model.BigLama()
        ten2pil = transforms.ToPILImage()

        results = []

        for image, mask in zip(images, masks):
            print(f"\n=== 處理新圖像 (IMG版本) ===")

            ori_image = tensor2pil(image)
            print(f"原始圖像尺寸: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            print(f"填充後圖像張量形狀: {pt_image.shape}")
            print(f"填充後圖像原始範圍: [{pt_image.min().item():.6f}, {pt_image.max().item():.6f}]")

            # 應用輸入歸一化
            pt_image = self.apply_input_normalization(pt_image, input_normalization)

            mask = mask.movedim(0, -1).movedim(0, -1)
            ori_mask = ten2pil(mask)
            ori_mask = ori_mask.convert('L')
            print(f"原始遮罩尺寸: {ori_mask.size}")

            p_mask = padmask(ori_mask)

            if p_mask.size != p_image.size:
                print("調整遮罩尺寸")
                p_mask = p_mask.resize(p_image.size)

            if not invert_mask:
                p_mask = ImageOps.invert(p_mask)

            p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)

            pt_mask = pil2tensor(gray)
            print(f"處理後遮罩張量形狀: {pt_mask.shape}")
            print(f"處理後遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            if pt_mask.max() > 1.0:
                pt_mask = pt_mask / 255.0

            print(f"歸一化後遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            pt_image = pt_image.float()
            pt_mask = pt_mask.float()

            print(f"輸入模型前 - 圖像: {pt_image.shape}, 遮罩: {pt_mask.shape}")
            print(f"輸入模型前 - 圖像範圍: [{pt_image.min().item():.6f}, {pt_image.max().item():.6f}]")
            print(f"輸入模型前 - 遮罩範圍: [{pt_mask.min().item():.6f}, {pt_mask.max().item():.6f}]")

            result = mylama(pt_image, pt_mask)

            print(f"模型輸出形狀: {result.shape}")
            print(f"模型輸出原始範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")
            print(f"模型輸出數據類型: {result.dtype}")

            if result.dim() == 4:
                result = result.squeeze(0)
            elif result.dim() == 3 and result.shape[0] == 1:
                result = result.squeeze(0)

            result = result.cpu().float()
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

            print(f"清理後輸出範圍: [{result.min().item():.6f}, {result.max().item():.6f}]")

            result = self.apply_output_denormalization(result, output_denormalization)

            try:
                img_result = ten2pil(result)
                print(f"成功轉換為 PIL 圖像，尺寸: {img_result.size}")
            except Exception as e:
                print(f"轉換 PIL 圖像時發生錯誤: {e}")
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)
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