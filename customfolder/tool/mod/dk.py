from libdk.darknet import *
from mod.util import vprint
import cv2

def gen_file_train_val(detections, image, colors, image_path, class_name):
    txtname = image_path.split('.')
    with open(txtname[0]+"."+txtname[1]+'.txt', 'w+') as f:
        for label, confidence, bbox in detections:
            left, top, right, bottom = bbox2points(bbox)

            xo = (right+left)/(2*image.shape[1])
            yo = (bottom+top)/(2*image.shape[0])
            wo = (right-left)/image.shape[1]
            ho = (bottom-top)/image.shape[0]   
                
            vprint([str(class_name.index(label))+" ",str(round(xo,6))+" ",str(round(yo,6))+" ",str(round(wo,6))+" ",str(round(ho,6)),'\n'])
            f.writelines([str(class_name.index(label))+" ",str(round(xo,6))+" ",str(round(yo,6))+" ",str(round(wo,6))+" ",str(round(ho,6)),'\n'])

            cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[label], 2)        
    return image

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = network_width(network)
    height = network_height(network)
    darknet_image = make_image(width, height, 3)

    #image = cv2.imread(image_path)
    image = image_path
    src = cv2.imread(image)
    image_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    preprocess_image = image_resized
    
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, thresh=thresh)
    free_image(darknet_image)
    return detections, preprocess_image