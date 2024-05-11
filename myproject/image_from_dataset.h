#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"
#include "detection.h"

// include my project functions
#include "functions.h"

// for vscode work
#ifndef IMAGES_PATH
#define IMAGES_PATH "briaDataSet"
#endif



class ImageFromDataset{

    std::vector<cv::Rect> groundThruthBoundingRect;
    std::vector<Detection> detections;

    DET_LABEL label;
    cv::Mat img_in;


};