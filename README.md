# Simple Vehicle Detection and Tracking

Simple Vehicle Detection and Tracking an application for videos of urban traffic, employing YOLO v2 implemented in OpenCV and a custom tracker algorithm.

## Compilation

The file `main.cpp` contains the code. For running the code, the video file must be in a folder and the path must be included **in the code** in the variable `folderName`, and in a subfolder called `YOLO`, the following files, obtained from YOLO Darknet website [1]:
- coco.names
- yolov2.cfg
- [yolov2.weights ](https://pjreddie.com/media/files/yolov2.weights)

References:

[1] Redmon, Joseph and Farhadi, Ali, "YOLO: Real-Time Object Detection", Available at: https://pjreddie.com/darknet/yolov2/
