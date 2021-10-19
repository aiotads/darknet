# Folder structure
```
darknet/customfolder
.
├── backup
│   ├── wilson_yolov4-custom_1000.weights
│   ├── wilson_yolov4-custom_2000.weights
│   ├── wilson_yolov4-custom_3000.weights
│   ├── wilson_yolov4-custom_best.weights
│   └── wilson_yolov4-custom_last.weights
├── USB_Dataset
│   ├── all_photos
│   │   ├── image.png
│   │   ├── image.txt
│   │   ├──     .
│   │   ├──     .
│   │   ├── classes.txt
│   ├── splitdata.py
│   ├── train.txt
│   └── val.txt
├── convert.sh
├── obj.data
├── obj.names
├── README.md
├── wilson_yolov4-custom.cfg
└── yolov4.conv.137

```
- backup : backup the training weight(when you training the weight will generate here)
- USB_Dataset : prepare the dataset you want to train here
    - all_photos : include all image data and label data
    - splitdata.py : the tool to help you split your training data to train.txt and val.txt
    - train.txt : absolute path image to train
    - val.txt : absolute path image to train
- convert.sh : the tool to help you convert image .bmp to .png
- obj.data : all the path information yolo need
- obj.name : the class name yolo need
- wilson_yolov4-custom.cfg : our custom darknet cfg(it can be convert to xmodel)
- yolov4.conv.137 : the pretrain model by yolo official
# Training
```
./darknet detector train customfolder/obj.data customfolder/wilson_yolov4-custom.cfg customfolder/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map -gpus 0
```
# Predict
- Video
    ```
    python darknet_video.py --input /workspace/wilson/data_defect.mp4 --weights customfolder/backup/wilson_yolov4-custom_3000.weights --config_file customfolder/wilson_yolov4-custom.cfg --data_file customfolder/obj.data
    ```
- Image
    ```
    python darknet_images.py --input /workspace/wilson/photo.png --weights customfolder/backup/wilson_yolov4-custom_3000.weights --config_file customfolder/wilson_yolov4-custom.cfg --data_file customfolder/obj.data
    ```