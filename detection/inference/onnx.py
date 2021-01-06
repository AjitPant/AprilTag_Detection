import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("model_unet.onnx")
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: np.random.randn(1,3,1024, 1024).astype(np.float32)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)
