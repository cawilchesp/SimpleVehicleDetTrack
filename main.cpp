#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main()
{
    // ----------------------------------- Settings ------------------------------------
    // Archivo
//    string folderName = "D:/OneDrive/Educacion/PUJ/Doctorado/Apps/Muestras/Videos"; // Windows Cs
//    string folderName = "C:/Users/cawil/OneDrive/Educacion/PUJ/Doctorado/Apps/Muestras/Videos"; // Windows Tab
    string folderName = "/home/doctorado/Muestras/Videos"; // Linux U
//    string folderName = "/home/andres/Muestras/Videos"; // Linux Cs
    string videoName = "jueves_23_2.avi";

    // Colores
    Scalar blue(255,0,0);
    Scalar green(0,150,0);
    Scalar red(0,0,255);
    Scalar yellow(0,200,200);
    Scalar purple(200,0,200);

    // ---------------------------------- Input video ----------------------------------
    string filepath = folderName + "/" + videoName;
    Mat currentFrame;
    int frame = 0;
    VideoCapture cap(filepath);
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    double fps = cap.get(CV_CAP_PROP_FPS);
    double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Frames per second : " << fps << endl;
    cout << "Frame count : " << frameCount << endl;
    namedWindow("Video", CV_WINDOW_AUTOSIZE);

    // ---------------------------------- Output video ----------------------------------
    VideoWriter outputVideo;
    Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    outputVideo.open(folderName + "/jueves_23_2_YOLO_Track.avi", cap.get(CAP_PROP_FOURCC), fps, S, true);

    // --------------------------------- YOLO Settings ----------------------------------
    String modelConfiguration = folderName + "/YOLO/yolov2.cfg";
    String modelBinary = folderName + "/YOLO/yolov2.weights";
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    vector<String> classNamesVec;
    string classNames = folderName + "/YOLO/coco.names";
    ifstream classNamesFile(classNames.c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    // ------------------------------------- Video --------------------------------------
    int cntOld = 0;
    vector<vector<int>> newObj;
    vector<int> dataNew(11);
    vector<vector<int>> oldObj;
    vector<Ptr<Tracker>> trackers;
    vector<Ptr<Tracker>> trackers_temp;
    int cntCar = 0, cntBus = 0, cntTruck = 0, cntMoto = 0, cntBike = 0;
    int upperLimit = 700;
    int lowerLimit = 20;

    while(cap.read(currentFrame)) {
        double frameMsec = cap.get(CV_CAP_PROP_POS_MSEC);
        frame++;
        cv::line(currentFrame,cv::Point(lowerLimit,0),cv::Point(lowerLimit,480),Scalar(0,0,255),1,8,0);
        cv::line(currentFrame,cv::Point(upperLimit,0),cv::Point(upperLimit,480),Scalar(0,0,255),1,8,0);

        // Detecciones en el cuadro actual
        int cntNew = 0;
        newObj.clear();

        // -------------------------------- YOLO OpenCV ----------------------------------
        if (currentFrame.channels() == 4)
            cvtColor(currentFrame, currentFrame, COLOR_BGRA2BGR);
        Mat inputBlob = blobFromImage(currentFrame, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data");                   //set the network input
        Mat detectionMat = net.forward("detection_out");   //compute output
        float confidenceThreshold = 0.05;
        for (int i = 0; i < detectionMat.rows; i++) {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
            if (confidence > confidenceThreshold) {
                float x_center = detectionMat.at<float>(i, 0) * currentFrame.cols;
                float y_center = detectionMat.at<float>(i, 1) * currentFrame.rows;
                float width = detectionMat.at<float>(i, 2) * currentFrame.cols;
                float height = detectionMat.at<float>(i, 3) * currentFrame.rows;

                if (x_center - width / 2 > lowerLimit && x_center + width / 2 < upperLimit && (objectClass == 1 || objectClass == 2 || objectClass == 3 || objectClass == 5 || objectClass == 7)) {
                    dataNew[0] = frame;
                    dataNew[1] = objectClass; // class
                    dataNew[2] = confidence; // score
                    dataNew[3] = x_center - width / 2; // left
                    dataNew[4] = y_center - height / 2; // top
                    dataNew[5] = x_center + width / 2; // right
                    dataNew[6] = y_center + height / 2; // bottom
                    dataNew[7] = x_center; //cc_x
                    dataNew[8] = y_center; //cc_y
                    dataNew[9] = 0; // count
                    dataNew[10] = 0; // label
                    newObj.push_back(dataNew);
                }
            }
        }

        cntNew = newObj.size();

        // Caso 1: Todos los objetos en la escena son nuevos
        if (cntOld == 0 && cntNew != 0) {
            trackers.clear();
            for(int b = 0; b < cntNew; b++) {
                trackers.push_back(TrackerKCF::create());
                Rect2d bbox(newObj[b][3], newObj[b][4], newObj[b][5] - newObj[b][3], newObj[b][6] - newObj[b][4]);
                trackers.back()->init(currentFrame, bbox);
                trackers.back()->update(currentFrame, bbox);
                rectangle(currentFrame, bbox, Scalar(0,0,0), 2, 8, 0);
                newObj[b][9] = 1;
            }
        } else

        // Caso 2: Todos los objetos en la escena se perdieron
        if (cntOld != 0 && cntNew == 0) {
            for(int c = 0; c < (int)trackers.size(); c++) {
                Rect2d bbox;
                trackers[c]->update(currentFrame, bbox);
                if (bbox.x + bbox.width < upperLimit && bbox.x > lowerLimit) {
                    Rect2d prevBox;
                    for(int j = 0; j < cntOld; j++) {
                        prevBox = Rect2d(oldObj[j][3], oldObj[j][4], oldObj[j][5] - oldObj[j][3], oldObj[j][6] - oldObj[j][4]);
                        Rect2d interBox = bbox & prevBox;
                        double p1 = interBox.area() / bbox.area();
                        double p2 = interBox.area() / prevBox.area();
                        if (p1 > 0.6 && p2 > 0.6) {
                            dataNew[0] = frame;
                            dataNew[1] = oldObj[j][1]; // class
                            dataNew[2] = oldObj[j][2]; // score
                            dataNew[3] = (int)bbox.x; // left
                            dataNew[4] = (int)bbox.y; // top
                            dataNew[5] = (int)bbox.x + (int)bbox.width; // right
                            dataNew[6] = (int)bbox.y + (int)bbox.height; // bottom
                            dataNew[7] = (int)bbox.x + ((int)bbox.width / 2); //cc_x
                            dataNew[8] = (int)bbox.y + ((int)bbox.height / 2); //cc_y
                            dataNew[9] = oldObj[j][9] + 1; // count

                            if (dataNew[9] == 2) {
                                if (oldObj[j][1] == 2) {
                                    cntCar++;
                                    dataNew[10] = cntCar; // label
                                } else if (oldObj[j][1] == 5) {
                                    cntBus++;
                                    dataNew[10] = cntBus; // label
                                } else if (oldObj[j][1] == 3) {
                                    cntMoto++;
                                    dataNew[10] = cntMoto; // label
                                } else if (oldObj[j][1] == 1) {
                                    cntBike++;
                                    dataNew[10] = cntBike; // label
                                } else if (oldObj[j][1] == 7) {
                                    cntTruck++;
                                    dataNew[10] = cntTruck; // label
                                }
                            } else {
                                dataNew[10] = oldObj[j][10]; // label
                            }

                            newObj.push_back(dataNew);
                            cntNew++;

                            Scalar colorBox;
                            string textLabel;
                            if (dataNew[1] == 2) {
                                textLabel = "Car ";
                                colorBox = blue;
                            } else if (dataNew[1] == 5) {
                                textLabel = "Bus ";
                                colorBox = green;
                            } else if (dataNew[1] == 3) {
                                textLabel = "Moto ";
                                colorBox = red;
                            } else if (dataNew[1] == 1) {
                                textLabel = "Bike ";
                                colorBox = yellow;
                            } else if (dataNew[1] == 7) {
                                textLabel = "Truck ";
                                colorBox = purple;
                            }

                            textLabel = textLabel + to_string(dataNew[10]);
                            Rect2d labelBox(bbox.x, bbox.y - 19, 100, 20);
                            rectangle(currentFrame, labelBox, colorBox, CV_FILLED, 8, 0);
                            putText(currentFrame, textLabel, Point2d(bbox.x+3, bbox.y-3), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255), 2);
                            rectangle(currentFrame, bbox, colorBox, 2, 8, 0);
                            break;
                        }
                    }
                } else {
                    trackers[c].~Ptr();
                }
            }
        } else

        // Caso 3: Algunos objetos se mantienen, otros se perdieron
        if (cntOld != 0 && cntNew != 0) {

            // eliminación de trackers que fallan al actualizar
            for(int c = 0; c < (int)trackers.size(); c++) {
                Rect2d bbox;
                bool okTracker = trackers[c]->update(currentFrame, bbox);
                if (okTracker == true) {
                    trackers_temp.push_back(trackers[c]);
                }
            }
            trackers = trackers_temp;
            trackers_temp.clear();

            // Creación de nuevos trackers a partir de nuevas detecciones
            int currentTrackers = trackers.size();
            vector<vector<int>> new_temp;
            int new_temp_cnt = 0;
            for(int b = 0; b < cntNew; b++) {
                Rect2d newBox(newObj[b][3], newObj[b][4], newObj[b][5] - newObj[b][3], newObj[b][6] - newObj[b][4]);
                bool newMatch = false;
                for(int c = 0; c < currentTrackers; c++) {
                    Rect2d bbox;
                    trackers[c]->update(currentFrame, bbox);
                    Rect2d interBox = bbox & newBox;
                    double p1 = interBox.area() / bbox.area();
                    double p2 = interBox.area() / newBox.area();
                    if (p1 > 0.6 || p2 > 0.6) {
                        newMatch = true;
                        break;
                    }
                }
                if (newMatch == false) {
                    trackers.push_back(TrackerKCF::create());
                    Rect2d detectBox(newObj[b][3], newObj[b][4], newObj[b][5] - newObj[b][3], newObj[b][6] - newObj[b][4]);
                    trackers.back()->init(currentFrame, detectBox);
                    trackers.back()->update(currentFrame, detectBox);
                    rectangle(currentFrame, detectBox, Scalar(0,0,0), 2, 8, 0);

                    dataNew[0] = frame;
                    dataNew[1] = newObj[b][1]; // class
                    dataNew[2] = newObj[b][2]; // score
                    dataNew[3] = (int)detectBox.x; // left
                    dataNew[4] = (int)detectBox.y; // top
                    dataNew[5] = (int)detectBox.x + (int)detectBox.width; // right
                    dataNew[6] = (int)detectBox.y + (int)detectBox.height; // bottom
                    dataNew[7] = (int)detectBox.x + ((int)detectBox.width / 2); //cc_x
                    dataNew[8] = (int)detectBox.y + ((int)detectBox.height / 2); //cc_y
                    dataNew[9] = 1; // count

                    new_temp.push_back(dataNew);
                    new_temp_cnt++;
                }
            }

            // Correlación de trackers con detecciones en t-1 y t
            currentTrackers = trackers.size();
            for(int c = 0; c < currentTrackers; c++) {
                Rect2d bbox;
                trackers[c]->update(currentFrame, bbox);
                if (bbox.x + bbox.width < upperLimit && bbox.x > lowerLimit) {
                    Rect2d prevBox;
                    for(int j = 0; j < cntOld; j++) {
                        prevBox = Rect2d(oldObj[j][3], oldObj[j][4], oldObj[j][5] - oldObj[j][3], oldObj[j][6] - oldObj[j][4]);
                        Rect2d interBox1 = bbox & prevBox;
                        double p11 = interBox1.area() / bbox.area();
                        double p12 = interBox1.area() / prevBox.area();
                        if (p11 > 0.6 && p12 > 0.6) {
                            dataNew[0] = frame;
                            dataNew[1] = oldObj[j][1]; // class
                            dataNew[2] = oldObj[j][2]; // score
                            dataNew[3] = (int)bbox.x; // left
                            dataNew[4] = (int)bbox.y; // top
                            dataNew[5] = (int)bbox.x + (int)bbox.width; // right
                            dataNew[6] = (int)bbox.y + (int)bbox.height; // bottom
                            dataNew[7] = (int)bbox.x + ((int)bbox.width / 2); //cc_x
                            dataNew[8] = (int)bbox.y + ((int)bbox.height / 2); //cc_y
                            dataNew[9] = oldObj[j][9] + 1; // count

                            if (dataNew[9] == 2) {
                                if (oldObj[j][1] == 2) {
                                    cntCar++;
                                    dataNew[10] = cntCar; // label
                                } else if (oldObj[j][1] == 5) {
                                    cntBus++;
                                    dataNew[10] = cntBus; // label
                                } else if (oldObj[j][1] == 3) {
                                    cntMoto++;
                                    dataNew[10] = cntMoto; // label
                                } else if (oldObj[j][1] == 1) {
                                    cntBike++;
                                    dataNew[10] = cntBike; // label
                                } else if (oldObj[j][1] == 7) {
                                    cntTruck++;
                                    dataNew[10] = cntTruck; // label
                                }
                            } else {
                                dataNew[10] = oldObj[j][10]; // label
                            }

                            new_temp.push_back(dataNew);
                            new_temp_cnt++;

                            Scalar colorBox;
                            string textLabel;
                            if (dataNew[1] == 2) {
                                textLabel = "Car ";
                                colorBox = blue;
                            } else if (dataNew[1] == 5) {
                                textLabel = "Bus ";
                                colorBox = green;
                            } else if (dataNew[1] == 3) {
                                textLabel = "Moto ";
                                colorBox = red;
                            } else if (dataNew[1] == 1) {
                                textLabel = "Bike ";
                                colorBox = yellow;
                            } else if (dataNew[1] == 7) {
                                textLabel = "Truck ";
                                colorBox = purple;
                            }

                            textLabel = textLabel + to_string(dataNew[10]);
                            Rect2d labelBox(bbox.x, bbox.y - 19, 100, 20);
                            rectangle(currentFrame, labelBox, colorBox, CV_FILLED, 8, 0);
                            putText(currentFrame, textLabel, Point2d(bbox.x+3, bbox.y-3), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255), 2);
                            rectangle(currentFrame, bbox, colorBox, 2, 8, 0);
                        }
                    }
                } else {
                    trackers[c].~Ptr();
                }
            }
            cntNew = new_temp_cnt;
            newObj = new_temp;
        }

        // Ordenar trackers
        if ((int)trackers.size() > 0) {
            for(int h = 0; h < (int)trackers.size(); h++) {
                if ( trackers[h] != 0 ) {
                    trackers_temp.push_back(trackers[h]);
                }
            }
            trackers = trackers_temp;
            trackers_temp.clear();
        }

        cntOld = cntNew;
        oldObj = newObj;

        // Time
        double totalSec = frameMsec / 1000;
        double totalMin = totalSec / 60;
        int min = (int)totalMin;
        int sec = (int)((totalMin - (double)min) * 60);
        int decmin = min / 10;
        min = min - (decmin * 10);
        int decsec = sec / 10;
        sec = sec - (decsec * 10);

        // Presentación del texto
        rectangle(currentFrame, Rect2d(0,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-30,(int)cap.get(CAP_PROP_FRAME_WIDTH), 30), Scalar(255,255,255), CV_FILLED, 8, 0);
        string textFrame, carText, busText, motoText, truckText, bikeText;
        textFrame = to_string(decmin) + to_string(min) + ":" + to_string(decsec) + to_string(sec);
        putText(currentFrame, textFrame, Point2d(10,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, Scalar(0,0,0), 2, LINE_8, false);
        carText = "Cars: " + to_string(cntCar);
        putText(currentFrame, carText, Point2d(80,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, blue, 2, LINE_8, false);
        busText = "Buses: " + to_string(cntBus);
        putText(currentFrame, busText, Point2d(190,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, green, 2, LINE_8, false);
        motoText = "Motos: " + to_string(cntMoto);
        putText(currentFrame, motoText, Point2d(310,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, red, 2, LINE_8, false);
        truckText = "Trucks: " + to_string(cntTruck);
        putText(currentFrame, truckText, Point2d(430,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, purple, 2, LINE_8, false);
        bikeText = "Bikes: " + to_string(cntBike);
        putText(currentFrame, bikeText, Point2d(560,(int)cap.get(CAP_PROP_FRAME_HEIGHT)-10), FONT_HERSHEY_PLAIN, 1.1, yellow, 2, LINE_8, false);

        imshow("Video", currentFrame);

        outputVideo.write(currentFrame); // Save video file

        if(waitKey(30) == 27)
            break;
    }
    return 0;
}
