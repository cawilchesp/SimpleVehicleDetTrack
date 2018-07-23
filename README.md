## Binaries for Simple Vehicle Detection and Tracking V 1.0

Simple Vehicle Detection and Tracking an application for videos of urban traffic, employing YOLO v2 implemented in OpenCV and a custom tracker algorithm.

Copyright (c) 2018 Carlos Wilches, Julian Quiroga
 
Pontificia Universidad Javeriana, Bogotá Colombia

### Terms of use

This program is provided for research purposes only. Any commercial use is prohibited. If you are interested in a commercial use, please  contact the copyright holder. 
 
This program is distributed WITHOUT ANY WARRANTY.

### Install

1. Install OpenCV 3.3.1 or above, with contrib modules (For DNN libraries with YOLO v2 implemented)
2. Compile `main.cpp`.

### Usage

1. **In the code**, the path of the folder containing the video file must be specified in the variable `folderName`.
2. **In the code**, the video file must be specified in the variable `videoName`.
3. In the **folder** specified above, a subfolder called `YOLO` must be created, containing the following files, obtained from YOLO v2 website [1]:

- coco.names
- yolov2.cfg
- [yolov2.weights ](https://pjreddie.com/media/files/yolov2.weights)

References:

[1] Redmon, Joseph and Farhadi, Ali, "YOLO: Real-Time Object Detection", Available at: https://pjreddie.com/darknet/yolov2/

### Bugs and Comments

Please report to Carlos Wilches (c.wilches@javeriana.edu.co)
