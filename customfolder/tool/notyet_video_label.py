import argparse
import os
import glob
from posixpath import join
import random
import time
import cv2
import numpy as np
import darknet


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    #image = cv2.imread(image_path)
    image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors,image_path,class_names)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
    
config_file = "/home/nvidia/wilson/new_darknet/darknet/customfolder/wilson_yolov4-custom.cfg"
data_file = "/home/nvidia/wilson/new_darknet/darknet/customfolder/obj.data"
weights = "/home/nvidia/wilson/new_darknet/darknet/customfolder/backup/wilson_yolov4-custom_3000.weights"

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    config_file,
    data_file,
    weights,
    batch_size=1
)
# -----------------Video bbox to img-----------------
path = "/workspace/wilson/"
input_video = os.path.join(path, 'data_perfect.mp4')
cap = cv2.VideoCapture(input_video)
pic_name_count = 0
while True:
    ret,frame = cap.read()
    image, detections = image_detection(
        frame, network, class_names, class_colors, thresh=0.5      
    )   
    darknet.print_detections(detections)
    pic_name_count = pic_name_count + 1
    cv2.imshow('Inference-'+str(pic_name_count), image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
# -----------------Video bbox to img-----------------