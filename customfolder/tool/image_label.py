import os
# from customfolder.libdk.darknet import class_colors, load_network
# import darknet
from mod.dk import *
from mod.util import *

def main():
    args = parser()
    check_arguments_errors(args)

    network, class_names, class_colors = load_network(args.config_file, args.data_file, args.weights)

    with open(os.path.join(args.image_path, 'classes.txt'), 'w+') as f:
        for class_name in class_names:
            f.writelines(class_name + '\n')

    for filename in os.listdir(args.image_path):
        if filename.endswith('.png'):
            images_path = os.path.join(args.image_path, filename)
            detections, preprocess_image = image_detection(images_path, network, class_names, class_colors, thresh=args.thresh)
            gen_file_train_val(detections , preprocess_image, class_colors, images_path, class_name)
            vprint(images_path)
            vprint_detections(detections)
    print('Smart labeling done!')

if __name__ == '__main__':
    main()