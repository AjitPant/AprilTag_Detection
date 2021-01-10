import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("model_unet.onnx")
from onnx_tf.backend import prepare

onnx_model = model 
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("saved.pb")  # export the model

