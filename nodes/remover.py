from PIL import ImageOps, ImageFilter
import torch
import numpy as np
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

    def _normalize_image(self, image_tensor):
        """
        將圖像張量歸一化到 [0, 1] 範圍
        TensorRT 模型期望輸入在 [0, 1] 範圍內
        """
        if image_tensor.max() > 1.0:
            # 如果圖像在 [0, 255] 範圍內，歸一化到 [0, 1]
            image_tensor = image_tensor / 255.0

        # 確保在 [0, 1] 範圍內
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        return image_tensor

    def _normalize_mask(self, mask_tensor):
        """
        將遮罩張量歸一化到 [0, 1] 範圍
        確保遮罩為單通道，並且數值在正確範圍內
        """
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0

        # 確保在 [0, 1] 範圍內
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        # 確保遮罩是單通道的
        if mask_tensor.dim() == 4 and mask_tensor.shape[1] != 1:
            # 如果是多通道，取第一個通道
            mask_tensor = mask_tensor[:, :1, :, :]
        elif mask_tensor.dim() == 3:
            # 如果是 3D，增加通道維度
            mask_tensor = mask_tensor.unsqueeze(1)

        return mask_tensor

    def _denormalize_output(self, output_tensor):
        """
        將輸出張量從 [0, 1] 範圍反歸一化，準備用於顯示
        """
        # 確保輸出在 [0, 1] 範圍內
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        return output_tensor

    def _prepare_tensors_for_trt(self, image_tensor, mask_tensor):
        """
        為 TensorRT 推理準備張量
        確保正確的形狀、設備和數據類型
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 確保張量在正確的設備上
        image_tensor = image_tensor.to(device, dtype=torch.float32)
        mask_tensor = mask_tensor.to(device, dtype=torch.float32)

        # 確保是 4D 張量 (batch, channels, height, width)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

        # 歸一化
        image_tensor = self._normalize_image(image_tensor)
        mask_tensor = self._normalize_mask(mask_tensor)

        # 調試資訊
        print(f"TRT 輸入圖像範圍: [{image_tensor.min().item():.4f}, {image_tensor.max().item():.4f}]")
        print(f"TRT 輸入遮罩範圍: [{mask_tensor.min().item():.4f}, {mask_tensor.max().item():.4f}]")
        print(f"TRT 輸入圖像形狀: {image_tensor.shape}")
        print(f"TRT 輸入遮罩形狀: {mask_tensor.shape}")

        return image_tensor, mask_tensor

    def lama_remover(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        # 初始化 TensorRT 模型（只初始化一次以提高效率）
        if not hasattr(self, '_mylama'):
            self._mylama = model.BigLama()
            print("✅ TensorRT 模型已初始化")

        ten2pil = transforms.ToPILImage()
        results = []

        for i, (image, mask) in enumerate(zip(images, masks)):
            print(f"\n--- 處理第 {i + 1} 張圖像 ---")

            ori_image = tensor2pil(image)
            print(f"原始圖像尺寸: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

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

            # 遮罩閾值處理
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            # 為 TensorRT 準備張量
            try:
                trt_image, trt_mask = self._prepare_tensors_for_trt(pt_image, pt_mask)

                # TensorRT 推理
                print("🚀 開始 TensorRT 推理...")
                with torch.no_grad():
                    result = self._mylama(trt_image, trt_mask)

                # 處理輸出
                result = self._denormalize_output(result)

                # 移除批次維度（如果存在）
                if result.dim() == 4 and result.shape[0] == 1:
                    result = result.squeeze(0)

                print(f"TRT 輸出範圍: [{result.min().item():.4f}, {result.max().item():.4f}]")
                print(f"TRT 輸出形狀: {result.shape}")

                # 轉換為 PIL 圖像
                img_result = ten2pil(result)

                # 裁剪到原始尺寸
                x, y = img_result.size
                if x > w or y > h:
                    img_result = cropimage(img_result, w, h)

                # 轉換為 ComfyUI 格式
                i = pil2comfy(img_result)
                results.append(i)

                print(f"✅ 第 {i + 1} 張圖像處理完成")

            except Exception as e:
                print(f"❌ TensorRT 推理失敗: {e}")
                print(f"錯誤詳情: {type(e).__name__}")

                # 錯誤處理：返回原始圖像
                print("🔄 返回原始圖像作為錯誤處理")
                ori_result = pil2comfy(ori_image)
                results.append(ori_result)

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

    def _normalize_image(self, image_tensor):
        """
        將圖像張量歸一化到 [0, 1] 範圍
        """
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        return image_tensor

    def _normalize_mask(self, mask_tensor):
        """
        將遮罩張量歸一化到 [0, 1] 範圍
        """
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        if mask_tensor.dim() == 4 and mask_tensor.shape[1] != 1:
            mask_tensor = mask_tensor[:, :1, :, :]
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        return mask_tensor

    def _denormalize_output(self, output_tensor):
        """
        將輸出張量反歸一化
        """
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        return output_tensor

    def _prepare_tensors_for_trt(self, image_tensor, mask_tensor):
        """
        為 TensorRT 推理準備張量
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        image_tensor = image_tensor.to(device, dtype=torch.float32)
        mask_tensor = mask_tensor.to(device, dtype=torch.float32)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

        image_tensor = self._normalize_image(image_tensor)
        mask_tensor = self._normalize_mask(mask_tensor)

        print(f"TRT 輸入圖像範圍: [{image_tensor.min().item():.4f}, {image_tensor.max().item():.4f}]")
        print(f"TRT 輸入遮罩範圍: [{mask_tensor.min().item():.4f}, {mask_tensor.max().item():.4f}]")

        return image_tensor, mask_tensor

    def lama_remover_IMG(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        if not hasattr(self, '_mylama'):
            self._mylama = model.BigLama()
            print("✅ TensorRT 模型已初始化")

        ten2pil = transforms.ToPILImage()
        results = []

        for i, (image, mask) in enumerate(zip(images, masks)):
            print(f"\n--- 處理第 {i + 1} 張圖像 (IMG 模式) ---")

            ori_image = tensor2pil(image)
            print(f"原始圖像尺寸: {ori_image.size}")

            w, h = ori_image.size
            p_image = padimage(ori_image)
            pt_image = pil2tensor(p_image)

            mask = mask.movedim(0, -1).movedim(0, -1)
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

            # 遮罩閾值處理
            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            # 為 TensorRT 準備張量
            try:
                trt_image, trt_mask = self._prepare_tensors_for_trt(pt_image, pt_mask)

                print("🚀 開始 TensorRT 推理...")
                with torch.no_grad():
                    result = self._mylama(trt_image, trt_mask)

                result = self._denormalize_output(result)

                if result.dim() == 4 and result.shape[0] == 1:
                    result = result.squeeze(0)

                print(f"TRT 輸出範圍: [{result.min().item():.4f}, {result.max().item():.4f}]")

                img_result = ten2pil(result)

                x, y = img_result.size
                if x > w or y > h:
                    img_result = cropimage(img_result, w, h)

                i = pil2comfy(img_result)
                results.append(i)

                print(f"✅ 第 {i + 1} 張圖像處理完成")

            except Exception as e:
                print(f"❌ TensorRT 推理失敗: {e}")
                ori_result = pil2comfy(ori_image)
                results.append(ori_result)

        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover (TensorRT)",
    "LamaRemoverIMG": "Big lama Remover(IMG) (TensorRT)"
}