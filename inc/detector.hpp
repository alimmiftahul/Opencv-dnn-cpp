/*
 * Filename: Opencv-dnn-cpp/inc/detector.hpp
 * Author: Alim
 * 
 * Copyright (c) 2022 Your Company
 */

#ifndef __DETECTOR_HPP
#define __DETECTOR_HPP

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>
#include <ctime>

using namespace cv;
using namespace dnn;
using namespace std;


class Detector
{
    private :
        struct Point2D{
                double X;
                double Y;
        };
        Point2D Postopr,
                Postopl,
                Posbottomr,
                Posbottoml,
                Posbottom,
                Posrobot,
                m_goalPos;


        VideoCapture cap;
        String  m_modelConfiguration,
                m_modelWeights;
        Net     m_net;
        Mat     m_frame;
        vector<string> m_clasess;
        float   m_confThreshold;
        float   m_nmsThreshold;
        Mat m_blob;
        string m_WinName;
        vector<Mat> m_outs;
        vector<double> m_layersTimes;
        double  m_freq;
        double  m_time;
        string  m_label,
                m_clasessFile,
                m_line;
        int     m_InpWidth ;  // Width of network's input image
        int     m_InpHeight ; 
        vector<String> names;
        int     frameCounter ;
        int     tick ;
        int     fps;
        int     m_objectcount,
                m_objectcountMax;
        int     mode;
        time_t  timeBegin;
        time_t  timeNow ;
        bool    m_goalbottomr,
                m_goalbottoml,
                m_goaltopr,
                m_goaltopl,
                m_detectbottom,
                m_robot,
                m_goal;

        

    public :
        Detector();
        void PostProcess(Mat& frame, const vector<Mat>& out);
        int DrawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
        vector<String> GetOutputsNames(const Net& net);
        int EndProcess();
        int Process();
        double Tracker(bool Object);
        double GetGoalX();
        double GetGoalY();
        bool GetGoal();
        void DisplayFps(Mat fps_frame);
        int Init();
};

#endif