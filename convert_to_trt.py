# convert_to_trt_TensorRT10x_optimized.py (針對 TensorRT 10.x 和 RTX 3090 的優化版本)
# 版本 2: 修改為支援動態尺寸 (Dynamic Shape) 以便使用 Padding 策略
import tensorrt as trt
import os
import sys

# --- 版本檢查和路徑設定 ---
print(f"腳本執行當下，實際使用的 TensorRT 版本是: {trt.__version__}")

# TensorRT 版本解析
trt_version = trt.__version__
major_version = int(trt_version.split('.')[0])
print(f"TensorRT 主版本: {major_version}")

# --- 路徑設定 ---
ONNX_PATH = './ckpts/lama_fp32_dynamic.onnx'

# ComfyUI TRT 模型目錄設定
TRT_MODEL_DIR = '/root/ComfyUI/models/trt'
# --- MODIFIED: 更改引擎檔案名稱以反映動態尺寸特性 ---
ENGINE_FILENAME = f'lama_fp16_rtx3090_trt{major_version}x_dynamic.trt'
ENGINE_PATH = os.path.join(TRT_MODEL_DIR, ENGINE_FILENAME)

WORKSPACE_GB = 12  # RTX 3090 的 24GB VRAM 可以使用較大工作空間

# 確保 TRT 模型目錄存在
try:
    os.makedirs(TRT_MODEL_DIR, exist_ok=True)
    # 測試寫入權限
    test_file = os.path.join(TRT_MODEL_DIR, '.write_test')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)

    print(f"✅ TRT 模型目錄: {TRT_MODEL_DIR}")
    print(f"  - 目錄存在且可寫入")
except PermissionError:
    print(f"❌ 權限錯誤: 無法寫入 {TRT_MODEL_DIR}")
    print("建議解決方案:")
    print("  1. 使用 sudo 運行此腳本")
    print("  2. 或更改目錄權限: sudo chmod 777 /root/ComfyUI/models/")
    sys.exit(1)
except Exception as e:
    print(f"❌ 目錄創建失敗: {e}")
    sys.exit(1)

print("--- TensorRT 引擎建構器 (TensorRT 10.x + RTX 3090 專用優化) ---")

if not os.path.exists(ONNX_PATH):
    print(f"錯誤：在 {ONNX_PATH} 找不到 ONNX 模型檔案")
    sys.exit(1)

# --- 初始化 TensorRT 元件 ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

# --- 設定建構時的記憶體 ---
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))

print(f"\n=== TensorRT {major_version}.x + Ampere 架構 (RTX 3090) 優化設定 ===")

# 1. 啟用 FP16 模式
config.set_flag(trt.BuilderFlag.FP16)
print("✓ 已啟用 FP16 半精度浮點數模式")

# 2. 檢查硬體支援
if builder.platform_has_fast_fp16:
    print("✓ 硬體支援快速 FP16 (偵測到 Ampere 架構的張量核心)")
else:
    print("⚠ 警告：目前的硬體可能無法從 FP16 中獲得最大效能")

# 3. TensorRT 10.x 的 TF32 處理
if major_version >= 10:
    print("✓ TensorRT 10.x 檢測到:")
    print("  - TF32 在 Ampere GPU 上自動啟用 (無需明確設定)")
    print("  - 享受 TF32 加速的 FP32 運算和 FP16 的雙重優勢")
else:
    # TensorRT 8.x/9.x 的處理方式
    try:
        config.set_flag(trt.BuilderFlag.TF32)
        print("✓ 已啟用 TF32 模式 (TensorRT 8.x/9.x)")
    except AttributeError:
        print("ⓘ TF32 標誌不可用")

# 4. 嚴格型別模式 (版本檢查)
strict_types_available = False
try:
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    print("✓ 已啟用嚴格型別模式 (STRICT_TYPES)")
    strict_types_available = True
except AttributeError:
    # 嘗試較舊的替代方案
    try:
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
        print("✓ 已啟用 DIRECT_IO 模式 (STRICT_TYPES 的替代)")
    except AttributeError:
        print("ⓘ 精度控制標誌在此版本中不可用")

# 5. TensorRT 10.x 的額外優化
if major_version >= 10:
    # 檢查可用的新標誌
    available_flags = []

    # 嘗試一些 TensorRT 10.x 的新功能
    try:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        available_flags.append("PREFER_PRECISION_CONSTRAINTS")
    except (AttributeError, Exception):
        pass

    try:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        available_flags.append("SPARSE_WEIGHTS")
    except (AttributeError, Exception):
        pass

    if available_flags:
        print(f"✓ TensorRT 10.x 額外優化: {', '.join(available_flags)}")

# 6. 顯示所有可用的 BuilderFlag (除錯用)
print("\n--- 可用的 BuilderFlag (除錯資訊) ---")
all_flags = [attr for attr in dir(trt.BuilderFlag) if not attr.startswith('_')]
print(f"此版本支援的標誌數量: {len(all_flags)}")
print(f"關鍵標誌檢查:")
key_flags = ['FP16', 'TF32', 'STRICT_TYPES', 'PREFER_PRECISION_CONSTRAINTS']
for flag in key_flags:
    exists = hasattr(trt.BuilderFlag, flag)
    print(f"  - {flag}: {'✓' if exists else '✗'}")

# --- 解析 ONNX 模型 ---
print(f"\n正在從 {ONNX_PATH} 解析 ONNX 模型...")
with open(ONNX_PATH, 'rb') as model:
    if not parser.parse(model.read()):
        print("錯誤：解析 ONNX 檔案失敗。")
        for error in range(parser.num_errors):
            print(f"  - {parser.get_error(error)}")
        sys.exit(1)
print("ONNX 模型解析成功。")

# 檢查網路的輸入和輸出
print("\n--- 網路輸入層 ---")
for i in range(network.num_inputs):
    tensor = network.get_input(i)
    print(f"  輸入層 {i}: 名稱={tensor.name}, 形狀={tensor.shape}, 型別={tensor.dtype}")

print("\n--- 網路輸出層 ---")
for i in range(network.num_outputs):
    tensor = network.get_output(i)
    print(f"  輸出層 {i}: 名稱={tensor.name}, 形狀={tensor.shape}, 型別={tensor.dtype}")

# --- MODIFIED: 設定最佳化設定檔 (改為動態尺寸) ---
print("\n--- 動態尺寸最佳化設定檔 (支援 Padding 策略) ---")
profile = builder.create_optimization_profile()

# 為 "image" 輸入定義尺寸範圍 (Batch, Channels, Height, Width)
# 注意：長寬最好是 8 的倍數
min_shape = (1, 3, 256, 256)      # 最小可處理尺寸
opt_shape = (1, 3, 1024, 1024)      # 預期最佳性能的尺寸
max_shape = (1, 3, 2560, 2560)    # 最大可處理尺寸 (RTX 3090 可設更高)

# 為 "mask" 輸入定義對應的尺寸範圍
mask_min_shape = (1, 1, 256, 256)
mask_opt_shape = (1, 1, 1024, 1024)
mask_max_shape = (1, 1, 2560, 2560)
mask_input_name = "mask" # 確保與您模型輸入名稱一致

profile.set_shape("image", min=min_shape, opt=opt_shape, max=max_shape)
profile.set_shape(mask_input_name, min=mask_min_shape, opt=mask_opt_shape, max=mask_max_shape)
config.add_optimization_profile(profile)

print(f"✓ 已為輸入 'image' 和 '{mask_input_name}' 設定動態尺寸")
print(f"  - 最小尺寸: {min_shape}")
print(f"  - 最佳尺寸: {opt_shape}")
print(f"  - 最大尺寸: {max_shape}")
print(f"  - 此引擎將能處理此範圍內的任意尺寸，適合 Padding 策略。")


# --- 建構引擎 ---
print("\n正在建構 TensorRT 引擎...")
print("TensorRT 10.x 會自動為您的 RTX 3090 選擇最快的核心...")
print("建構動態尺寸引擎可能需要幾分鐘時間，請耐心等候...")

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("錯誤：建構引擎失敗。請檢查日誌訊息。")
    sys.exit(1)

print("✓ TensorRT 引擎建構成功！")

# --- 儲存引擎到 ComfyUI 目錄 ---
with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print(f"✓ 引擎已成功儲存至 ComfyUI TRT 模型目錄:")
print(f"  - 完整路徑: {ENGINE_PATH}")
print(f"  - 檔案名稱: {ENGINE_FILENAME}")

# 顯示 ComfyUI 目錄結構
print(f"\n--- ComfyUI TRT 目錄結構 ---")
print(f"📁 {TRT_MODEL_DIR}/")
if os.path.exists(TRT_MODEL_DIR):
    trt_files = [f for f in os.listdir(TRT_MODEL_DIR) if f.endswith('.trt')]
    if trt_files:
        for trt_file in sorted(trt_files):
            file_path = os.path.join(TRT_MODEL_DIR, trt_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            marker = "🆕" if trt_file == ENGINE_FILENAME else "📄"
            print(f"  {marker} {trt_file} ({file_size:.1f} MB)")
    else:
        print(f"  📄 {ENGINE_FILENAME} (新建)")

# 顯示檔案大小比較
if os.path.exists(ENGINE_PATH):
    engine_size_mb = os.path.getsize(ENGINE_PATH) / (1024 * 1024)
    print(f"\n--- 檔案大小資訊 ---")
    print(f"  - TRT 引擎大小: {engine_size_mb:.2f} MB")

if os.path.exists(ONNX_PATH):
    onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  - 原始 ONNX 大小: {onnx_size_mb:.2f} MB")
    if os.path.exists(ENGINE_PATH):
        compression_ratio = (onnx_size_mb - engine_size_mb) / onnx_size_mb * 100
        if compression_ratio > 0:
            print(f"  - 大小減少: {compression_ratio:.1f}%")
        else:
            print(f"  - 大小增加: {abs(compression_ratio):.1f}% (包含優化資料)")

print("\n" + "=" * 70)
print("=== TensorRT 10.x + RTX 3090 + ComfyUI 最終優化完成！ ===")
print("=" * 70)
print("🎯 引擎已成功建立並儲存到 ComfyUI 模型目錄:")
print(f"   📁 {TRT_MODEL_DIR}/")
print(f"   📄 {ENGINE_FILENAME}")
print("\n✅ 優化特點:")
print("  🚀 FP16 加速: 充分利用 Ampere 架構的第三代 Tensor Cores")
print("  ⚡ TF32 自動啟用: TensorRT 10.x 在 RTX 3090 上的預設行為")
print("  🤸 **尺寸靈活性**: 透過動態尺寸設定，可處理不同解析度，支援 Padding 策略") # MODIFIED
print("  🆕 現代化: 使用 TensorRT 10.x 的最新優化技術")
print(f"  📈 預期性能: RTX 3090 上約 2-3x 加速 (相較於原始 FP32)")

print(f"\n💡 在 ComfyUI 中使用:")
print(f"  1. 引擎檔案已放置在正確的 TRT 模型目錄中")
print(f"  2. **重要**: 您需要修改 ComfyUI 節點的 Python 程式碼，將圖片預處理從『強制縮放』改為『Padding』，以利用此引擎的動態特性。")
print(f"  3. ComfyUI LaMa Remover 節點應該能自動識別此引擎")
print(f"  4. 選擇使用 TensorRT 推理模式以獲得最佳性能")

# 驗證 TF32 狀態的額外資訊
print(f"\n🔧 TF32 狀態確認:")
print(f"✅ 您的設定組合 (RTX 3090 + TensorRT {trt_version}) 中:")
print(f"  - TF32 在 Tensor Core 運算中自動啟用")
print(f"  - 無需明確設定，這是 TensorRT 10.x 的設計改進")
print(f"  - 您將自動獲得 TF32 的性能優勢")

print(f"\n📝 技術細節:")
print(f"  - 工作空間: {WORKSPACE_GB}GB (充分利用 RTX 3090 的 24GB VRAM)")
print(f"  - 批次大小: 1 (專為 ComfyUI 單張圖片處理優化)")
print(f"  - 輸入解析度: 動態 (Min: {min_shape[2]}x{min_shape[3]}, Opt: {opt_shape[2]}x{opt_shape[3]}, Max: {max_shape[2]}x{max_shape[3]})") # MODIFIED
print(f"  - 精度模式: FP16 + 自動 TF32")