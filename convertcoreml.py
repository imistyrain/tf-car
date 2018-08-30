import urllib, os, sys, zipfile
from os.path import dirname
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
tf_model_path ="car"+"/" + 'frozen_inference_graph.pb'
with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
input_node_names = ['Preprocessor/sub']
output_node_names = ['concat', 'concat_1']
gdef = strip_unused_lib.strip_unused(
        input_graph_def = original_gdef,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = dtypes.float32.as_datatype_enum)
# Save the feature extractor to an output file
frozen_model_file = "ck"+"/" + 'ssd_mobilenet_feature_extractor.pb'
with gfile.GFile(frozen_model_file, "wb") as f:
    f.write(gdef.SerializeToString())
import tfcoreml
# Supply a dictionary of input tensors' name and shape (with # batch axis)
input_tensor_shapes = {"Preprocessor/sub:0":[1,256,256,3]} # batch size is 1
# Output CoreML model path
coreml_model_file = "ck"+"/" + 'ssd_mobilenet_feature_extractor.mlmodel'
# The TF model's ouput tensor name
output_tensor_names = ['concat:0', 'concat_1:0']

# Call the converter. This may take a while
coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names)