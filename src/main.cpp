/*
 * Filename: Opencv-dnn-cpp/src/main.cpp
 * Author: alim
 * 
 * Copyright (c) 2022 Your Company
 */

#include <detector.hpp>
int main()
{
    Detector detector;
    detector.Init();
    while(1)
    {
        
        detector.Process();
        detector.EndProcess();
    }
    return 0;

}