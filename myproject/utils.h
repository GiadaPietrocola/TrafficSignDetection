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
float min_circularity = 0.7f;
float sigma = 0.1f;
float min_width_perc = 0.08f; // minimum width of light expressed as percentage of image width
float max_width_perc = 0.8f;  // maximum width ...

float min_area_perc = 0.002f;
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

        int kernel_size = ucas::round(6 * sigma);
        if (kernel_size % 2 == 0)
            kernel_size += 1;

        cv::GaussianBlur(img_in, img_blurred, cv::Size(kernel_size, kernel_size), sigma);

        // cv::morphologyEx(img_blurred, img_blurred, cv::MORPH_CLOSE,
        //                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1)));

        cv::Canny(img_blurred, edges, 100 / 3, 100, 3, false);
    }
};
