#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"

// include my project functions
#include "functions.h"

// for vscode work
#ifndef IMAGES_PATH
#define IMAGES_PATH "briaDataSet"
#endif

class Detection{
    cv::Rect boundingRect;
    DET_LABEL label;
    float confidence_score;
    float iou;


};