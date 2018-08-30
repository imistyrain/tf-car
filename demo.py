#coding=utf-8
import sys,os,platform,cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
if platform.system()=="Windows":
    models_dir="D:/CNN/tenflowusage/models-master"
else:
    models_dir="/home/yanyu/models"
sys.path.append(models_dir+"/research")
sys.path.append(models_dir+"/research/object_detection")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

DATASET="car"
MODEL_FILE=DATASET+"/frozen_inference_graph.pb"
PATH_TO_LABELS = os.path.join(DATASET, DATASET+'_label_map.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
NUM_CLASSES =len(label_map.item)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def list_all_nodes(graph):
    with graph.as_default():
        for li in [op.values() for op in tf.get_default_graph().get_operations()]:
            print(li)
def load_graph(pbpath):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pbpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

def test_one_image(graph,image_path="test_images/table_8_3_790.jpg"):
    #image_np = load_image_into_numpy_array(Image.open(image_path))
    image_np=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    output_dict = run_inference_for_single_image(image_np, graph)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)
    #plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.show()

def test_dir(graph,dir="test_images/"):
    files=os.listdir(dir)
    for file in files:
        img_path=dir+"/"+file
        print(file)
        test_one_image(graph,img_path)

def test_camera(detection_graph,index=0):
    cap=cv2.VideoCapture(index)
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            image_np=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=6)
            cv2.imshow("capture", image_np)
            cv2.waitKey(1)

if __name__=="__main__":
    graph=load_graph(MODEL_FILE)
    #list_all_nodes(graph)
    test_dir(graph)
    #test_camera(graph)