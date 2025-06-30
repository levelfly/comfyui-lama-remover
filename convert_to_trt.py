# convert_to_trt_static_optimized.py (針對 RTX 3090 和固定單張圖片的最終優化版本)
import tensorrt as trt
import os

# --- 加入這行來驗證 ---
print(f"腳本執行當下，實際使用的 TensorRT 版本是: {trt.__version__}")
# --- 驗證結束 ---

# --- 設定 ---
ONNX_PATH = './ckpts/lama_fp32.onnx'
# 檔名清楚標示其為 RTX 3090 專用、靜態 (static) 且批次為 1 (bs1)
ENGINE_PATH = './ckpts/lama_fp16_rtx3090_static_bs1.trt'
# RTX 3090 有 24GB VRAM，可設定較大的工作空間以利 TensorRT 搜尋最佳演算法
WORKSPACE_GB = 12

print("--- TensorRT 引擎建構器 (針對 RTX 3090 和固定單張圖片的最終優化) ---")
print(f"目前 TensorRT 版本: {trt.__version__}")

if not os.path.exists(ONNX_PATH):
    print(f"錯誤：在 {ONNX_PATH} 找不到 ONNX 模型檔案")
    exit()

# --- 初始化 TensorRT 元件 ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 設定日誌記錄器等級
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

# --- 設定建構時的記憶體與精度 ---

# 設定建構時的最大工作空間記憶體
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))

print("\n=== Ampere 架構 (RTX 3090) 優化設定 ===")

# 1. 啟用 FP16 模式：這是效能提升和記憶體降低的主要來源
config.set_flag(trt.BuilderFlag.FP16)
print("✓ 已啟用 FP16 半精度浮點數模式")

# 2. 檢查硬體是否支援快速 FP16 (利用 Tensor Cores)
if builder.platform_has_fast_fp16:
    print("✓ 硬體支援快速 FP16 (偵測到 Ampere 架構的張量核心)")
else:
    print("⚠ 警告：目前的硬體可能無法從 FP16 中獲得最大效能")

# 3. 啟用 TF32 模式 (針對 Ampere 架構的補充優化)
# 說明：TF32 是 Ampere GPU 的預設模式，用於加速 FP32 運算。
#       在啟用 FP16 模式後，此設定主要影響那些無法轉換為 FP16 的層。
#       此處為顯式設定，確保其開啟。
try:
    config.set_flag(trt.BuilderFlag.TF32)
    print("✓ 已啟用 TF32 模式，可加速殘留的 FP32 運算")
except AttributeError:
    print("ⓘ 此 TensorRT 版本不支援 TF32 旗標 (通常在較舊版本中)")

# 4. 啟用嚴格型別模式 (STRICT_TYPES)
# 說明：此旗標會要求 TensorRT 嚴格遵守層的資料型別，避免非預期的精度轉換。
#       這有助於獲得更穩定和可預測的結果，特別是在使用 FP16 時。
try:
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    print("✓ 已啟用嚴格型別模式 (Strict Types)")
except AttributeError:
    print("ⓘ 此 TensorRT 版本不支援 STRICT_TYPES 旗標")


# --- 解析 ONNX 模型 ---
print(f"\n正在從 {ONNX_PATH} 解析 ONNX 模型...")
with open(ONNX_PATH, 'rb') as model:
    if not parser.parse(model.read()):
        print("錯誤：解析 ONNX 檔案失敗。")
        for error in range(parser.num_errors):
            print(f"  - {parser.get_error(error)}")
        exit()
print("ONNX 模型解析成功。")

# 檢查網路的輸入和輸出 (除錯用)
print("\n--- 網路輸入層 ---")
for i in range(network.num_inputs):
    tensor = network.get_input(i)
    print(f"  輸入層 {i}: 名稱={tensor.name}, 形狀={tensor.shape}, 型別={tensor.dtype}")

print("\n--- 網路輸出層 ---")
for i in range(network.num_outputs):
    tensor = network.get_output(i)
    print(f"  輸出層 {i}: 名稱={tensor.name}, 形狀={tensor.shape}, 型別={tensor.dtype}")


# --- 設定最佳化設定檔 (Optimization Profile) ---
# [關鍵修改] 為了針對您的 ComfyUI API (一次一張圖) 獲得最快速度，
# 我們將設定檔從動態改為靜態，將 min, opt, max 全部設為 1。
profile = builder.create_optimization_profile()

# 定義固定尺寸 (批次大小為 1)
fixed_shape = (1, 3, 512, 512)
mask_shape = (1, 1, 512, 512)
mask_input_name = "mask"

# 將 'image' 的 min, opt, max 全都設定為固定尺寸
profile.set_shape("image", min=fixed_shape, opt=fixed_shape, max=fixed_shape)
# 將 'mask' 的 min, opt, max 全都設定為固定尺寸
profile.set_shape(mask_input_name, min=mask_shape, opt=mask_shape, max=mask_shape)

config.add_optimization_profile(profile)

print("\n--- 靜態尺寸最佳化設定檔 ---")
print(f"✓ 已為輸入 'image' 和 '{mask_input_name}' 設定靜態批次 (Batch Size = 1)")
print(f"  - 此引擎被高度特化，專為處理單張 512x512 圖片，以達到最低延遲。")


# --- 建構並序列化引擎 ---
print("\n正在建構 TensorRT 引擎... 這可能會需要數分鐘，請耐心等候。")
print("TensorRT 會自動為您的 RTX 3090 和固定尺寸選擇最快的核心 (kernel)...")

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("錯誤：建構引擎失敗。請檢查日誌訊息。")
    exit()

print("✓ TensorRT 引擎建構成功！")


# --- 儲存引擎到檔案 ---
with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print(f"✓ 引擎已成功儲存至: {ENGINE_PATH}")

# 顯示檔案大小
if os.path.exists(ENGINE_PATH):
    engine_size_mb = os.path.getsize(ENGINE_PATH) / (1024 * 1024)
    print(f"  - 引擎檔案大小: {engine_size_mb:.2f} MB")
if os.path.exists(ONNX_PATH):
    onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  - 原始 ONNX 大小: {onnx_size_mb:.2f} MB")


print("\n==============================================")
print("=== RTX 3090 單張圖片處理 - 最終優化完成！ ===")
print("==============================================")
print("這個引擎已針對您的硬體和使用情境進行以下優化:")
print("  ✓ FP16 & TF32 加速: 充分利用 Ampere 架構的計算能力")
print("  ✓ 最低延遲: 透過靜態尺寸設定，將性能完全集中在單張圖片處理")
print("  ✓ 穩定性: 啟用嚴格型別模式，確保結果可預測")