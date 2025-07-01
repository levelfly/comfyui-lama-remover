# -*- coding: utf-8 -*-
# Filename: export_dynamic_onnx.py
# Description: A standalone script to export the LaMa inpainting model to a dynamic shape ONNX file.
# Based on: https://colab.research.google.com/github/Carve-Photos/lama/blob/main/export_LaMa_to_onnx.ipynb

import os
import sys
import subprocess
import torch
import yaml
import requests
import zipfile
import io
from omegaconf import OmegaConf


def install_dependencies():
    """Install required Python packages."""
    print("檢查並安裝必要的套件...")
    packages = [
        "omegaconf",
        "webdataset",
        "pytorch_lightning",
        "kornia==0.5.0",
        "onnx",
        "onnxruntime"
    ]
    for package in packages:
        try:
            # 使用 __import__ 檢查套件是否存在
            __import__(package.split('==')[0])
            print(f"套件 {package} 已安裝。")
        except ImportError:
            print(f"正在安裝套件: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("所有依賴套件均已準備就緒。")


def download_and_unzip_model():
    """Download and unzip the Big-LaMa model checkpoint."""
    model_dir = "big-lama"
    zip_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
    ckpt_path = os.path.join(model_dir, "models", "best.ckpt")

    if os.path.exists(ckpt_path):
        print(f"模型檔案 {ckpt_path} 已存在，跳過下載。")
        return

    print(f"正在從 {zip_url} 下載模型...")
    response = requests.get(zip_url)
    response.raise_for_status()  # Will raise an error for bad status codes

    print("正在解壓縮模型...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(".")

    if os.path.exists(ckpt_path):
        print("模型下載並解壓縮成功。")
    else:
        raise FileNotFoundError("解壓縮後未找到模型檔案，請檢查 ZIP 檔案內容。")


# --- Model Loading and Exporting Logic ---
# This part is adapted directly from the Colab notebook.
try:
    # 嘗試從已安裝的套件導入，這是標準做法
    from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
except ImportError:
    # 如果失敗，表示可能需要從本地 clone 的 repo 導入
    print("警告: 無法從標準路徑導入 'saicinpainting'。正在嘗試從本地 'lama' 目錄導入。")
    print("請確保您已執行 'git clone https://github.com/Carve-Photos/lama.git --depth 1'")
    sys.path.append(os.path.abspath('lama'))
    from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


class ExportLama(torch.nn.Module):
    """A wrapper module for exporting the LaMa model with pre- and post-processing."""

    def __init__(self, lama_model):
        super().__init__()
        self.model = lama_model

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        # The model expects inputs in [0, 1] range
        masked_img = image * (1 - mask)

        if self.model.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        predicted_image = self.model.generator(masked_img)
        inpainted = mask * predicted_image + (1 - mask) * image

        # The notebook clamps the output to [0, 255] range
        return torch.clamp(inpainted * 255, min=0, max=255)


def main():
    """Main function to run the export process."""
    # 1. Install dependencies
    install_dependencies()

    # 2. Download model checkpoint
    download_and_unzip_model()

    # 3. Load the configuration file
    print("正在載入模型配置...")
    config_path = "big-lama/config.yaml"
    with open(config_path, "r") as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # 4. Prepare model configuration for loading
    kwargs = dict(config.training_model)
    kwargs.pop("kind")
    kwargs["use_ddp"] = False  # Set to False for single device inference/export

    # Enable JIT version of FourierUnit, required for export
    config.generator.resnet_conv_kwargs.use_jit = True
    # Fix the configuration by setting the weight to zero
    if hasattr(config.losses, "resnet_pl"):
        config.losses.resnet_pl.weight = 0

    # 5. Load the model state from the checkpoint
    print("正在載入模型權重...")
    ckpt_path = "big-lama/models/best.ckpt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    lama_model = DefaultInpaintingTrainingModule(config, **kwargs)
    lama_model.load_state_dict(state["state_dict"], strict=False)
    lama_model.on_load_checkpoint(state)
    lama_model.freeze()
    lama_model.eval()
    print("模型載入成功。")

    # 6. Wrap the model for export
    exported_model = ExportLama(lama_model)
    exported_model.eval()
    exported_model.to("cpu")

    # 7. Export to ONNX format
    print("正在匯出至 ONNX 格式 (含動態尺寸)...")
    output_path = os.path.join("ckpts", "lama_fp32_dynamic.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        exported_model,
        (
            torch.rand(1, 3, 512, 512).type(torch.float32).to("cpu"),
            torch.rand(1, 1, 512, 512).type(torch.float32).to("cpu")
        ),
        output_path,
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
        verbose=False,
    )

    print("\n" + "=" * 50)
    print(f"✅ 匯出成功！")
    print(f"   動態尺寸的 ONNX 模型已儲存至: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

#cd /root/ComfyUI/custom_nodes/comfyui-lama-remover/onnx
#git clone https://github.com/Carve-Photos/lama.git --depth 1
#pip install webdataset
#pip install pytorch_lightning
#pip install "albumentations==0.4.6"
#python3.11 export_dynamic_onnx.py
#cd /root/ComfyUI/custom_nodes/comfyui-lama-remover
#python3.11 convert_to_trt.py