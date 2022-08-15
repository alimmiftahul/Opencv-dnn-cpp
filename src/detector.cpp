/*
 * Filename: /Opencv-dnn-cpp/src/detector.cpp
 * Author: Alim
 * 
 * Copyright (c) 2022 Your Company
 */


#include <detector.hpp>
#include <iostream>

int Detector::Init()
{
    m_confThreshold = 0.6; // Confidence threshold
    m_nmsThreshold = 0.4; ;
    m_clasessFile = "/home/rscuad/Documents/alim_skripsi/Opencv-dnn-cpp/Data/coco.names";
    ifstream ifs(m_clasessFile.c_str());
    while (getline(ifs, m_line)) m_clasess.push_back(m_line);
    m_modelConfiguration = "/home/rscuad/Documents/alim_skripsi/Opencv-dnn-cpp/Data/yolo4-tiny.cfg";
    m_modelWeights = "/home/rscuad/Documents/alim_skripsi/Opencv-dnn-cpp/Data/yolov4_goal.weights";

    m_net = readNetFromDarknet(m_modelConfiguration, m_modelWeights);

    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH ,640); // 320,640,1920
    cap.set(CAP_PROP_FRAME_HEIGHT,480);//240,480,1080
    m_WinName = "Deep learning object detection in OpenCV";
    namedWindow(m_WinName, WINDOW_NORMAL);
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video cam" << std::endl;
		return -1;
	}
    // fps count variabel
    timeBegin = std::time(0);
    frameCounter = 0;
    tick = 0;
    
}
int Detector::Process()
{
    cap>>m_frame;
    resize(m_frame, m_frame, Size(800, 600));
	if (m_frame.empty()) {
		std::cout << "image not found" << std::endl;
		return -1;
	}
    blobFromImage(m_frame, m_blob, 1/255.0, cv::Size(416, 416), Scalar(0,0,0), true, false);

    
    m_net.setInput(m_blob); //Sets the input to the network

    
    m_net.forward(m_outs, GetOutputsNames(m_net)); // Runs the forward pass to get output of the output layers

    /*
     fps algo
     */
    frameCounter++;
    timeNow = std::time(0) - timeBegin;
    if (timeNow - tick >= 1)
    {
        tick++;
        fps = frameCounter;
        frameCounter = 0;
    }

    PostProcess(m_frame, m_outs);
    /*
    time video
    */
    m_freq = getTickFrequency() / 1000;
    m_time = m_net.getPerfProfile(m_layersTimes) / m_freq;
    m_label = format("Inference time for a frame : %.2f ms", m_time);
    /*
    show
    */
    putText(m_frame, format("Average FPS=%d", fps ), Point(0, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
    putText(m_frame, m_label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    imshow(m_WinName, m_frame);
    // EndPrresocess();
}
int Detector::EndProcess()
{
     if (cv::waitKey(30) == 27 ) 
        return 0;
}
void Detector::DrawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = format("%.2f", conf);
    if (!m_clasess.empty())
    {
        CV_Assert(classId <(int)m_clasess.size());
        label = m_clasess[classId] + ":" + label;
        std::cout<<"class : "<<m_clasess[classId]<<std::endl;
        std::cout<<"x : "<<left<<std::endl;
        std::cout<<"y : "<<top<<std::endl;
        std::cout<<"-----------------------\n";

    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.65 , Scalar(0,0,0),1); //0.75
}
void Detector::PostProcess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the #include <iostream>highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > m_confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        DrawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }

}
vector<String> Detector::GetOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;

}

