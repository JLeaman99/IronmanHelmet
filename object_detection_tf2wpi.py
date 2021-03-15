
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pyautogui    
import ctypes

lock = False; 
pyautogui.FAILSAFE = False;


DATA_DIR = os.getcwd()
MODELS_DIR = 'exported'
MODEL_NAME = 'v3c2'
MISC = 'annotations'
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME,'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME,'pipeline.config'))

LABEL_FILENAME = 'labelmap.pbtxt'
PATH_TO_LABELS = os.path.join(MISC, LABEL_FILENAME)

# %%
# Load the model
# ~~~~~~~~~~~~~~

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

video = cv2.VideoCapture('http://raspberrypi:8080/?action=stream')
#video = cv2.VideoCapture('test2.mp4')
ret = video.set(3,1280)
ret = video.set(4,720)

while True:
    # Read frame from camera
    ret, image_np = video.read()

    image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np_rgb, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    frame = image_np_with_detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

   
    
    # Display output
    cv2.imshow('object detection', frame)

    if not lock: 
        cv2.imwrite("frame0.jpg", frame)
        cmd = "del frame1.jpg"
        os.system(cmd)
       #Push to Background
        ctypes.windll.user32.SystemParametersInfoW(20, 0, f"{os.getcwd()}/frame0.jpg", 0)
        lock = True
  
    else: 
        cv2.imwrite("frame1.jpg", frame)
        cmd = "del frame0.jpg"
        os.system(cmd)
       #Push to Background
        ctypes.windll.user32.SystemParametersInfoW(20, 0, f"{os.getcwd()}/frame1.jpg", 0)
        lock = False
    
    print("Printing Detections");
    #print([category_index.get(value) for index,value in enumerate(detections['detection_classes'][0].numpy())]);
    print(detections['detection_boxes'][0]);
    #if([category_index.get(value) for index,value in enumerate(detections['detection_classes'][0].numpy())] == 'finger')
    movex = detections['detection_boxes'][0][0][1].numpy()*1920;
    movey = detections['detection_boxes'][0][0][0].numpy()*1080;
    print(movex);
  
    pyautogui.moveTo(movex,movey);
    
    print("Detections Done");
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()








