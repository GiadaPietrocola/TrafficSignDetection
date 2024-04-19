// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// include my project functions
#ifndef EXAMPLE_IMAGES_PATH
#define EXAMPLE_IMAGES_PATH "example_images"
#endif

struct Utils
{

    static void RealCanny(cv::Mat &img_in, cv::Mat &img_blurred, cv::Mat &edges, int sigma)
    {
        cv::cvtColor(img_in, img_blurred, cv::COLOR_BGR2GRAY);

        int kernel_size = ucas::round(6 * sigma);
        if (kernel_size % 2 == 0)
            kernel_size++;

        cv::GaussianBlur(img_in, img_blurred, cv::Size(kernel_size, kernel_size), sigma);

        cv::Canny(img_blurred, edges, 100 / 3, 100, 3, false);
    }
};