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
├── tool
│   ├── convert.sh
│   ├── darknet.py
│   ├── image_label.py
│   ├── label.py
│   ├── libdarknet.so
│   ├── notyet_video_label.py
│   └── splitdata.py
├── USB_Dataset
│   └── all_photos
│       ├── image.png
│       ├── image.txt
│       ├──     .
│       ├──     .
│       └── classes.txt
├── train.txt
├── val.txt
├── convert.sh
├── obj.data
├── obj.names
├── README.md
├── wilson_yolov4-custom.cfg
└── yolov4.conv.137

```
- backup : backup the training weight(when you training the weight will generate here)
- tool : help us to prepare data
    - splitdata.py : split your training data to train.txt and val.txt
    - convert.sh : convert image .bmp to .png
    - image_label.py : use weight to label photo
- USB_Dataset : prepare the dataset you want to train here
    - all_photos : include all image data and label data
    > Can change dataset directry to other path, if your storage space is less.
    
- train.txt : absolute path image to train
- val.txt : absolute path image to train
- obj.data : all the path information yolo need
- obj.name : the class name yolo need
- wilson_yolov4-custom.cfg : our custom darknet cfg(it can be convert to xmodel)
- yolov4.conv.137 : the pretrain model by yolo official
# Training
## calculate anchor first (cmd for yoloV4)
```
./darknet detector calc_anchors customfolder/obj.data -num_of_clusters 9 -width 608 -height 608 -showpause
```
It will generate the anchors.txt in darknet/

change the anchor in .cfg and save.
## start to train
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
    Can add threshold parameter to show only confidence over then it 
    
    ex.
    ```
    python darknet_video.py --input /workspace/wilson/data_defect.mp4 --weights customfolder/backup/wilson_yolov4-custom_3000.weights --config_file customfolder/wilson_yolov4-custom.cfg --data_file customfolder/obj.data --thresh 0.7
    ```