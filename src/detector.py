######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from video_stream import VideoStream
import discord

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default="model/")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')


args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

debug=False

# =======================  START MODEL SETUP ============================

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       



# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


food_hit_box_ymin = 100
food_hit_box_xmin = 130
food_hit_box_ymax = 320
food_hit_box_xmax = 530
food_hit_box_width = food_hit_box_xmax - food_hit_box_xmin
food_hit_box_height = food_hit_box_ymax - food_hit_box_ymin

frame_count = 0
current_time = 0
last_message_time = 30
message_cooldown = 30
detection_count = 0

display_cooldown_message = True

#Mínimo de Frames com detecção em um intervalo de 5 sec
message_detection_count_threshold = 50

print ('Iniciando Detecção...')

while True:

    # Grab frame from video stream
    original_frame = videostream.read()

    frame_count += 1

    if frame_count % 12 == 0:
        current_time += 1
        # print(current_time)

    if frame_count % 150 == 0:
        display_cooldown_message = True
        detection_count = 0

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = original_frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    cv2.rectangle(frame, (food_hit_box_xmin, food_hit_box_ymin), (food_hit_box_xmax, food_hit_box_ymax), (255, 10, 0), 4)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):

        if(classes[i] != 16):
            continue

        if(scores[i] <= min_conf_threshold):
            continue

        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1,(boxes[i][0] * imH)))
        xmin = int(max(1,(boxes[i][1] * imW)))
        ymax = int(min(imH,(boxes[i][2] * imH)))
        xmax = int(min(imW,(boxes[i][3] * imW)))

        detected_height = ymax - ymin
        detected_width = xmax - xmin

        intersect_x =(abs((xmin +  detected_width/2) - (food_hit_box_xmin + food_hit_box_width/2)) * 2 < ( detected_width + food_hit_box_width))
        intersect_y =(abs((ymin +  detected_height/2) - (food_hit_box_ymin + food_hit_box_height/2)) * 2 < ( detected_height + food_hit_box_height))

        if not(intersect_x and intersect_y):
            continue

        detection_count += 1

        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        
        if debug:
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        if(detection_count > message_detection_count_threshold):

            if (current_time - last_message_time > message_cooldown):

                detection_count = 0
                last_message_time = current_time

                # print(ymin)
                # print(xmin)
                # print(ymax)
                # print(xmax)

                crop_img = frame[ ymin:ymax, xmin:xmax] 

                # cv2.imshow("frame",frame)
                # cv2.imshow("frame_resized",frame_resized)

                average = crop_img.mean(axis=0).mean(axis=0) 

                pixels = np.float32(crop_img.reshape(-1, 3))

                n_colors = 5
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS

                _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)

                dominant = palette[np.argmax(counts)]

                # print(palette)
                # print(dominant)
        
                best_color_distance = 99999
                min_color_distance = 80

                file_name = f'output/detected-cat.jpg'

                cv2.imwrite(file_name, original_frame)

                print('Enviando mensagem...')
                
                discord.send_message(f'Gato detectado!', file_name)

                print('Mensagem enviada!')

            else: 
                if display_cooldown_message:
                    print("Gato detectado, porém o envio de mensagem está em cooldown")
                    print(f'{current_time} - {last_message_time} > {message_cooldown}')
                    display_cooldown_message = False

    if debug:
        cv2.imshow('Object detector', frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
           break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
