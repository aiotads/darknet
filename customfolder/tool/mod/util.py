import argparse
import os
from libdk.darknet import print_detections

def vprint(str):
    args = parser()
    if args.verbose:
        print(str)

def vprint_detections(str):
    args = parser()
    if args.verbose:
        print_detections(str)

def parser():
    parser = argparse.ArgumentParser(description="Smart labeling")
    parser.add_argument("-ip", "--image_path", type=str, 
                        help="the data set directory")
    parser.add_argument("-df" ,"--data_file", default="./cfg/coco.data",
                        help="path to data file obj.data")
    parser.add_argument("-w", "--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("-cfg", "--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("-t", "--thresh", type=float, default=0.5,
                        help="remove detections with confidence below this value")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show more info")
    return parser.parse_args()
    
def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.image_path and not os.path.exists(args.image_path):
        raise(ValueError("Invalid image data path {}".format(os.path.abspath(args.image_path))))
