#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// include my project functions
#ifndef IMAGES_PATH
#define IMAGES_PATH "example_images"
#endif

// //------------------------------------------------------------------------------
// //======================= CONSTANTS =================================
float min_circularity = 0.6f;
float sigma = 1.5f;
float min_width_perc = 0.08f; // minimum width of light expressed as percentage of image width
float max_width_perc = 0.8f;  // maximum width ...

float min_area_perc = 0.0002f;
float max_area_perc = 0.5f;

// data structures
enum class Color
{
    UNKNOWN,
    GREEN,
    ORANGE,
    RED
};
struct Detection
{
    Color color;
    int frame;
    double goodness;

    Detection(Color _color, int _frame, double _goodness = 0)
    {
        color = _color;
        frame = _frame;
        goodness = _goodness;
    }
};

// utility degrees 2 radians (and viceversa) functions
static inline double rad2deg(double radians)
{
    return radians * (180.0 / ucas::PI);
}
static inline double deg2rad(double degrees)
{
    return degrees * (ucas::PI / 180.0);
}

struct Utils
{
    static cv::Mat histogramEqualization(cv::Mat img)
    {
        std::vector<int> hist = ucas::histogram(img);

        int L = 256;
        std::vector<int> hist_eq_LUT(L);
        float norm_factor = float(L - 1) / (img.rows * img.cols);
        int accum = 0;
        for (int k = 0; k < hist_eq_LUT.size(); k++)
        {
            accum += hist[k];
            hist_eq_LUT[k] = norm_factor * accum;
        }

        for (int y = 0; y < img.rows; y++)
        {
            unsigned char *yRow = img.ptr<unsigned char>(y);
            for (int x = 0; x < img.cols; x++)
                yRow[x] = hist_eq_LUT[yRow[x]];
        }

        return img;
    }

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

        int kernel_size = ucas::round(6 * sigma);
        if (kernel_size % 2 == 0)
            kernel_size += 1;

        std::vector<cv::Mat> img_channels(3);
        cv::split(img_in, img_channels);
        histogramEqualization(img_channels[0]);
        histogramEqualization(img_channels[1]);
        histogramEqualization(img_channels[2]);
        cv::Mat img_output_BGReq;
        cv::merge(img_channels, img_output_BGReq);

        ipa::imshow("Contrast", img_output_BGReq);

        //     cv::cvtColor(img_output_BGReq, img_output_BGReq, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(img_output_BGReq, img_blurred, cv::Size(kernel_size, kernel_size), sigma);
        /*

          //  double canny_thresh = cv::threshold(img_blurred, img_blurred, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            double canny_thresh = 50;


            // Convertiamo la matrice in un vettore
            std::vector<int> values;
            if (img_blurred.isContinuous()) {
                values.assign(img_blurred.data, img_blurred.data + img_blurred.total());
            }
            else {
                for (int i = 0; i < img_blurred.rows; ++i) {
                    values.insert(values.end(), img_blurred.ptr<int>(i), img_blurred.ptr<int>(i) + img_blurred.cols);
                }
            }

            // Ordiniamo il vettore
            std::sort(values.begin(), values.end());

            // Calcoliamo la mediana
            int median;
            size_t size = values.size();
            if (size % 2 == 0) {
                median = (values[size / 2 - 1] + values[size / 2]) / 2;
            }
            else {
                median = values[size / 2];
            }

            std::cout << median;
            int lower = int(std::max(0, int(0.7*median)));
            int upper = int(std::min(255, int(1.1 * median)));


           cv::Canny(img_blurred, edges,  lower, upper, 3, false);

            // A Close operation could be useful??
          //  cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
         //                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

              cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", edges);
              */
    }
};
