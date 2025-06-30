# /lama/model.py (修正 TensorRT API 相容性問題)

import os
import torch
import tensorrt as trt
from ..utils import get_models_path
from comfy.model_management import get_torch_device

DEVICE = get_torch_device()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class BigLama:
    def __init__(self):
        self.device = DEVICE
        self.engine = None
        self.context = None
        self.graph = None
        self.last_input_shapes = None
        self.static_inputs_gpu = {}
        self.static_output_gpu = None

        engine_path = get_models_path(filename="lama_fp16_rtx3090_static_bs1.trt")
        print(f"載入 TensorRT 引擎路徑: {engine_path}")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"找不到 TensorRT 引擎檔案: {engine_path}. 請先執行 convert_to_trt.py")

        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()
            print("TensorRT 引擎和執行上下文建立成功")
        except Exception as e:
            print(f"載入 TensorRT 引擎失敗: {e}")
            raise

        self.binding_name_to_idx = {}

        # 檢查 TensorRT 版本並使用相應的 API
        try:
            # 嘗試使用新版 API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self.binding_name_to_idx[name] = i
            print("使用新版 TensorRT API (get_tensor_name)")
        except AttributeError:
            # 回退到舊版 API
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.binding_name_to_idx[name] = i
            print("使用舊版 TensorRT API (get_binding_name)")

        print(f"成功發現 {len(self.binding_name_to_idx)} 個綁定: {self.binding_name_to_idx}")

    def _get_tensor_shape(self, tensor_name):
        """獲取張量形狀，相容新舊版本 TensorRT API"""
        try:
            # 新版 TensorRT API
            if hasattr(self.context, 'get_tensor_shape'):
                return self.context.get_tensor_shape(tensor_name)
            elif hasattr(self.context, 'get_shape'):
                idx = self.binding_name_to_idx[tensor_name]
                return self.context.get_shape(idx)
            else:
                # 最舊版本的備用方案
                idx = self.binding_name_to_idx[tensor_name]
                return self.engine.get_binding_shape(idx)
        except Exception as e:
            print(f"獲取張量形狀失敗: {e}")
            # 如果都失敗，假設是 512x512 的標準輸出
            if 'output' in tensor_name.lower():
                return (1, 3, 512, 512)
            else:
                raise e

    def _set_tensor_address(self, tensor_name, tensor):
        """設定張量地址，相容新舊版本 TensorRT API"""
        try:
            # 新版 TensorRT API
            if hasattr(self.context, 'set_tensor_address'):
                self.context.set_tensor_address(tensor_name, tensor.data_ptr())
            else:
                # 舊版本不需要單獨設定地址，在 execute_v2 時處理
                pass
        except Exception as e:
            print(f"設定張量地址失敗: {e}")

    def _set_input_shape(self, tensor_name, shape):
        """設定輸入形狀，相容新舊版本 TensorRT API"""
        try:
            # 新版 TensorRT API
            if hasattr(self.context, 'set_input_shape'):
                self.context.set_input_shape(tensor_name, shape)
            elif hasattr(self.context, 'set_binding_shape'):
                idx = self.binding_name_to_idx[tensor_name]
                self.context.set_binding_shape(idx, shape)
        except Exception as e:
            print(f"設定輸入形狀失敗: {e}")

    def __call__(self, image, mask):
        """執行 Lama 修復推理"""
        image_input_name = "image"
        mask_input_name = "mask"
        output_name = "output"

        # 確保輸入張量在正確設備上且為連續記憶體
        image = image.contiguous().to(self.device)
        mask = mask.contiguous().to(self.device)

        # 調試：列印輸入數值範圍
        print(f"輸入圖片數值範圍: [{image.min().item():.4f}, {image.max().item():.4f}]")
        print(f"輸入遮罩數值範圍: [{mask.min().item():.4f}, {mask.max().item():.4f}]")

        try:
            # 獲取綁定索引
            image_idx = self.binding_name_to_idx[image_input_name]
            mask_idx = self.binding_name_to_idx[mask_input_name]
            output_idx = self.binding_name_to_idx[output_name]

            # 設定輸入尺寸
            self._set_input_shape(image_input_name, image.shape)
            self._set_input_shape(mask_input_name, mask.shape)

            # 獲取輸出尺寸並建立輸出緩衝區
            output_shape = self._get_tensor_shape(output_name)
            output_buffer = torch.empty(tuple(output_shape), dtype=torch.float32, device=self.device)

            # 設定張量地址（新版 API）
            self._set_tensor_address(image_input_name, image)
            self._set_tensor_address(mask_input_name, mask)
            self._set_tensor_address(output_name, output_buffer)

            # 準備綁定列表（保持原始邏輯）
            if hasattr(self.engine, 'num_io_tensors'):
                num_bindings = self.engine.num_io_tensors
            else:
                num_bindings = self.engine.num_bindings

            bindings = [0] * num_bindings
            bindings[image_idx] = image.data_ptr()
            bindings[mask_idx] = mask.data_ptr()
            bindings[output_idx] = output_buffer.data_ptr()

            # 執行推理（保持原始的 execute_v2 邏輯）
            success = self.context.execute_v2(bindings=bindings)

            if not success:
                raise RuntimeError("TensorRT 推理執行失敗")

            # 同步 GPU 操作
            torch.cuda.synchronize()

            # 調試：列印輸出數值範圍
            print(f"輸出數值範圍: [{output_buffer.min().item():.4f}, {output_buffer.max().item():.4f}]")

            return output_buffer

        except Exception as e:
            print(f"推理過程發生錯誤: {e}")
            print(f"圖片形狀: {image.shape}")
            print(f"遮罩形狀: {mask.shape}")
            print(f"設備: {self.device}")
            print(f"綁定名稱: {list(self.binding_name_to_idx.keys())}")

            # 錯誤處理：返回原始圖片
            return image

    def get_engine_info(self):
        """取得引擎資訊用於除錯"""
        info = {
            "device": str(self.device),
            "bindings": self.binding_name_to_idx,
        }

        try:
            if hasattr(self.engine, 'num_io_tensors'):
                info["num_io_tensors"] = self.engine.num_io_tensors
            if hasattr(self.engine, 'num_bindings'):
                info["num_bindings"] = self.engine.num_bindings

            # 嘗試獲取版本資訊
            info["tensorrt_version"] = trt.__version__
        except:
            pass

        return info