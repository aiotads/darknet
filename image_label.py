import argparse
import os
import glob
from posixpath import join
import random
import time
import cv2
import numpy as np
import darknet

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    #image = cv2.imread(image_path)
    image = image_path
    src = cv2.imread(image)
    image_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.label_draw_boxes(detections, image_resized, class_colors,image_path,class_names)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
    
# -----------------Video bbox to img-----------------
# path = "/workspace/wilson/"
# input_video = os.path.join(path, 'data_perfect.mp4')
# cap = cv2.VideoCapture(input_video)
# pic_name_count = 0
# while True:
#     ret,frame = cap.read()
#     image, detections = image_detection(
#         frame, network, class_names, class_colors, thresh=0.5      
#     )   
#     darknet.print_detections(detections)
#     pic_name_count = pic_name_count + 1
#     cv2.imshow('Inference-'+str(pic_name_count), image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cap.destroyAllWindows()
# -----------------Video bbox to img-----------------



if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    print(type(args))
    print(args)

    # random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )

    # -----------------img to yolo label-----------------
    path = '/home/nvidia/wilson/new_darknet/darknet/customfolder/USB_Dataset/all_photos/'
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            print(filename)
            images = path+filename
            image, detections = image_detection(
                images, network, class_names, class_colors, thresh=0.5      
            )

            darknet.print_detections(detections)

        cv2.imshow('Inference-'+str(filename), image)
        # cv2.waitKey() # STOP per image until press 'ENTER' key
        cv2.destroyAllWindows()
    # -----------------img to yolo label-----------------