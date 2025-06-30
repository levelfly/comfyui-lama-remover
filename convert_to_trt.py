# convert_to_trt.py (Final version with correct input names)

import tensorrt as trt
import os

# --- 設定 ---
ONNX_PATH = './ckpts/lama_fp32.onnx'
ENGINE_PATH = './ckpts/lama_fp32.trt'
WORKSPACE_GB = 4

print("--- TensorRT Engine Builder (from ONNX) ---")

if not os.path.exists(ONNX_PATH):
    print(f"Error: ONNX model not found at {ONNX_PATH}")
    exit()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)
print(f"FP16 mode enabled: True")

print(f"Parsing ONNX model from {ONNX_PATH}...")
with open(ONNX_PATH, 'rb') as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
print("ONNX model parsed successfully.")

profile = builder.create_optimization_profile()
fixed_shape = (1, 3, 512, 512)

# [最終修正] 根據偵查結果，遮罩的輸入名稱是 "mask"
mask_input_name = "mask" 
mask_shape = (1, 1, 512, 512)

# 為 'image' 設定 profile
profile.set_shape("image", min=fixed_shape, opt=fixed_shape, max=fixed_shape)
# 為 'mask' 設定 profile
profile.set_shape(mask_input_name, min=mask_shape, opt=mask_shape, max=mask_shape)
config.add_optimization_profile(profile)
print(f"Optimization profile set for inputs: 'image', '{mask_input_name}' with fixed size 512x512.")

print("Building TensorRT engine... This may take several minutes.")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("Error: Failed to build the engine.")
    exit()

print("Engine built successfully!")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized_engine)

print(f"TensorRT engine saved to: {ENGINE_PATH}")
print("Conversion complete!")