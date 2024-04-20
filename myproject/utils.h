#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// include my project functions
#ifndef EXAMPLE_IMAGES_PATH
#define EXAMPLE_IMAGES_PATH "example_images"
#endif

// //------------------------------------------------------------------------------
// //======================= CONSTANTS =================================
float min_circularity = 0.6f;
float sigma = 0.1f;
float min_width_perc = 0.08f; // minimum width of light expressed as percentage of image width
float max_width_perc = 0.8f;  // maximum width ...

float min_area_perc = 0.0002f;
float max_area_perc = 0.5f;

struct Utils
{

    // //------------------------------------------------------------------------------
    // //======================= METHODS =================================

    /** @brief Real version of Canny's Algorithm
     @param img_in Input image
     @param img_blurred Output image after Gaussian Blur
     @param edges Output edge's image after Canny's algorithm
     @param sigma Sigma coefficient of Gaussian Blur
    */
    static void RealCanny(cv::Mat &img_in, cv::Mat &img_blurred, cv::Mat &edges, int sigma)
    {

        cv::cvtColor(img_in, img_blurred, cv::COLOR_BGR2GRAY);
        double canny_thresh = cv::threshold(img_blurred, img_blurred, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        int kernel_size = ucas::round(6 * sigma);
        if (kernel_size % 2 == 0)
            kernel_size += 1;

        cv::GaussianBlur(img_in, img_blurred, cv::Size(kernel_size, kernel_size), sigma);

        cv::Canny(img_blurred, edges, canny_thresh / 3, canny_thresh, 3, false);

        // A Close operation could be useful??
        cv::morphologyEx(edges, edges, cv::MORPH_DILATE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    }
};
