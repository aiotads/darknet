import argparse
import os
import cv2
import darknet

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
    image = darknet.label_draw_boxes(detections, image_resized, class_colors, image_path, class_names)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)

    # random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )

    with open(os.path.join(args.image_path, 'classes.txt'), 'w+') as f:
        for class_name in class_names:
            f.writelines(class_name + '\n')

    for filename in os.listdir(args.image_path):
        if filename.endswith('.png'):
            images = os.path.join(args.image_path, filename)
            image, detections = image_detection(
                images, network, class_names, class_colors, thresh=args.thresh      
            )
            if args.verbose:
                print(images)
                darknet.print_detections(detections)

    print('Smart labeling done!')