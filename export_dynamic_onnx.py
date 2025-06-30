# export_dynamic_onnx.py
import torch
import yaml
import os
import requests
from saic_vision.models.lama import LaMa

# --- 1. 下載必要的模型和配置文件 ---
print("正在下載 LaMa 模型和配置文件...")
# 創建目錄
os.makedirs("big-lama", exist_ok=True)
os.makedirs("big-lama/models", exist_ok=True)

# 下載配置文件
url_config = "https://raw.githubusercontent.com/Carve-Photos/lama/main/big-lama/config.yaml"
r = requests.get(url_config)
with open("big-lama/config.yaml", 'wb') as f:
    f.write(r.content)

# 下載模型權重
url_model = "https://huggingface.co/Carve/LaMa/resolve/main/big-lama.pt"
r = requests.get(url_model)
with open("big-lama/models/best.pth", 'wb') as f:
    f.write(r.content)

print("下載完成。")

# --- 2. 載入 PyTorch 模型 ---
print("正在載入 PyTorch 模型...")
# 讀取配置
with open('big-lama/config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 初始化模型
model = LaMa(train_config.model)
# 載入模型權重
checkpoint = torch.load('big-lama/models/best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("模型載入成功。")

# --- 3. 執行 ONNX 匯出 ---
print("正在匯出至 ONNX (含動態尺寸)...")
# 建立虛擬輸入
dummy_image = torch.randn(1, 3, 512, 512, device=device)
dummy_mask = torch.randn(1, 1, 512, 512, device=device)

input_names = ['image', 'mask']
output_names = ['output']
output_path = "./ckpts/lama_fp32_dynamic.onnx"

# 確保目標目錄存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

torch.onnx.export(
    model,
    (dummy_image, dummy_mask),
    output_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=14,
    dynamic_axes={
        'image': {0: 'batch_size', 2: 'height', 3: 'width'},
        'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
print(f"✅ 成功！動態尺寸的 ONNX 模型已儲存至: {output_path}")