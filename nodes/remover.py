# --- 基礎導入 ---
from PIL import ImageOps, ImageFilter
import torch
from torchvision import transforms

# --- ComfyUI 工具函式導入 ---
# 假設這些函式位於 ..utils 資料夾中
from ..utils import cropimage, padimage, padmask, tensor2pil, pil2tensor, cropimage, pil2comfy

# --- LaMa 模型導入 ---
from ..lama import model

# --- [新增] 嘗試匯入自訂 CUDA 核心 ---
try:
    # 嘗試導入 lama_cpp 的 CUDA 核心
    from lama_cpp import _C as custom_cuda_blur

    LAMA_CPP_AVAILABLE = True
    print("✅ 成功匯入 LaMa 自訂 CUDA 模糊核心。")
except ImportError:
    LAMA_CPP_AVAILABLE = False
    print("⚠️  未找到 LaMa 自訂 CUDA 模糊核心。模糊處理將使用 Pillow (CPU)。")

# --- [新增] 全局模型快取 (Global Model Cache) ---
# 這是實現單例模式的關鍵，確保所有節點共享一個模型實例，避免重複載入
MODEL_CACHE = {
    "model": None,
    "device": None
}


def get_shared_model():
    """
    獲取全局唯一的、已預熱的模型實例。
    所有節點都調用此函式，以確保它們共享同一個模型。
    """
    # 如果快取中沒有模型，則進行初始化
    if MODEL_CACHE["model"] is None:
        print("🔄 [模型管理器] 首次初始化 BigLama 模型...")

        # 實例化模型
        model_instance = model.BigLama()
        MODEL_CACHE["model"] = model_instance
        MODEL_CACHE["device"] = model_instance.device

        # --- 預熱機制 (Warm-up Mechanism) ---
        print("🔥 [模型管理器] 正在預熱 TensorRT 引擎... (首次運行可能需要一些時間)")
        try:
            # 創建與模型相同設備的 dummy 張量
            device = MODEL_CACHE["device"]
            dummy_image = torch.zeros((1, 3, 512, 512), device=device, dtype=torch.float32)
            dummy_mask = torch.zeros((1, 1, 512, 512), device=device, dtype=torch.float32)

            # 執行一次推理以觸發所有初始化和優化
            with torch.no_grad():
                _ = model_instance(dummy_image, dummy_mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # 確保 GPU 操作完成

            print("✅ [模型管理器] BigLama 模型已載入並預熱完畢，準備進行高速推理。")
        except Exception as e:
            print(f"❌ [模型管理器] 模型預熱期間發生錯誤: {e}")
            # 如果預熱失敗，重置快取以便下次重試
            MODEL_CACHE["model"] = None
            MODEL_CACHE["device"] = None
            raise e  # 拋出異常以通知使用者

    # 返回快取中的模型和設備
    return MODEL_CACHE["model"], MODEL_CACHE["device"]


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
        mylama, device = get_shared_model()
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

            if gaussblur_radius > 0:
                if LAMA_CPP_AVAILABLE and device.type == 'cuda':
                    mask_tensor = pil2tensor(p_mask).to(device)
                    blurred_mask_tensor = custom_cuda_blur.gaussian_blur(mask_tensor, gaussblur_radius)
                    p_mask = ten2pil(blurred_mask_tensor.cpu().squeeze(0))
                else:
                    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            result = mylama(pt_image, pt_mask)

            # --- [修正] 在這裡移除批次維度 ---
            img_result = ten2pil(result.squeeze(0))

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
        mylama, device = get_shared_model()
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

            if gaussblur_radius > 0:
                if LAMA_CPP_AVAILABLE and device.type == 'cuda':
                    mask_tensor = pil2tensor(p_mask).to(device)
                    blurred_mask_tensor = custom_cuda_blur.gaussian_blur(mask_tensor, gaussblur_radius)
                    p_mask = ten2pil(blurred_mask_tensor.cpu().squeeze(0))
                else:
                    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=gaussblur_radius))

            gray = p_mask.point(lambda x: 0 if x > mask_threshold else 255)
            pt_mask = pil2tensor(gray)

            result = mylama(pt_image, pt_mask)

            # --- [修正] 在這裡移除批次維度 ---
            img_result = ten2pil(result.squeeze(0))

            x, y = img_result.size
            if x > w or y > h:
                img_result = cropimage(img_result, w, h)

            i = pil2comfy(img_result)
            results.append(i)

        return (torch.cat(results, dim=0),)


# --- ComfyUI 節點註冊 ---
NODE_CLASS_MAPPINGS = {
    "LamaRemover": LamaRemover,
    "LamaRemoverIMG": LamaRemoverIMG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemover": "Big lama Remover (CUDA Optimized)",
    "LamaRemoverIMG": "Big lama Remover IMG (CUDA Optimized)"
}