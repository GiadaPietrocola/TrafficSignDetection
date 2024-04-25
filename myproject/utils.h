#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// for vscode work
#ifndef IMAGES_PATH
#define IMAGES_PATH "briaDataSet"
#endif

// //------------------------------------------------------------------------------
// //======================= CONSTANTS =================================
float min_circularity = 0.6f;
float sigma = 1.0f;
float min_width_perc = 0.08f; // minimum width of light expressed as percentage of image width
float max_width_perc = 0.8f;  // maximum width ...

float min_area_perc = 0.0002f;
float max_area_perc = 0.5f;

int minHueValue = 0;
int maxHueValue = 20;

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
        cv::cvtColor(img_in, img_in, cv::COLOR_BGR2GRAY);

        //  double canny_thresh = cv::threshold(img_blurred, img_blurred, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        double canny_thresh = 50;

        /*
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
*/

        cv::Canny(img_in, edges, canny_thresh, 3 * canny_thresh, 3, false);

        // A Close operation could be useful??
        //  cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
        //                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        //   cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", edges);
    }

    /**
     * @brief Function that compute the right kernel size and executes the GaussianBlur
     *
     * @param img_in
     * @param img_out
     * @param sigma
     */
    static void CustomGaussianBlur(cv::Mat &img_in, cv::Mat &img_out, float sigma)
    {
        int kernel_size = ucas::round(6 * sigma);
        if (kernel_size % 2 == 0)
            kernel_size += 1;
        cv::GaussianBlur(img_in, img_out, cv::Size(kernel_size, kernel_size), sigma);
    }

    /**
     * @brief Function that computes the Median value of
     *
     * @param channel
     * @return double
     */
    static double calculateMedian(const cv::Mat &channel)
    {
        // Create a vector to store pixel values
        std::vector<uchar> pixels;
        pixels.reserve(channel.rows * channel.cols);

        // Copy pixel values to the vector
        for (int y = 0; y < channel.rows; ++y)
        {
            for (int x = 0; x < channel.cols; ++x)
            {
                pixels.push_back(channel.at<uchar>(y, x));
            }
        }

        // Sort the pixel values
        std::sort(pixels.begin(), pixels.end());

        // Calculate the median
        if (pixels.size() % 2 == 0)
        {
            // If the number of pixels is even, take the average of the two middle values
            return static_cast<double>(pixels[pixels.size() / 2 - 1] + pixels[pixels.size() / 2]) / 2.0;
        }
        else
        {
            // If the number of pixels is odd, take the middle value
            return static_cast<double>(pixels[pixels.size() / 2]);
        }
    }

    /**
     * @brief Color Histogram Equalization Function
     *
     * @param img image input to equalize in place
     * @return cv::Mat output image
     */
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

    /** @brief Histogram Equalization on BGR channels
    @param img_in Input image
    @param img_blurred Output image after Histogram Equalization
    */
    static void HistogramBGReq(const cv::Mat &img_in, cv::Mat &img_output_BGReq)
    {
        std::vector<cv::Mat> img_channels(3);
        cv::split(img_in, img_channels);
        histogramEqualization(img_channels[0]);
        histogramEqualization(img_channels[1]);
        histogramEqualization(img_channels[2]);
        cv::merge(img_channels, img_output_BGReq);

        ipa::imshow("Contrast", img_output_BGReq);
    }

    static void HistogramLabeq(const cv::Mat &img_in, cv::Mat &img_output_Labeq)
    {
        // Convert the image to Lab color space
        cv::Mat labImage;
        cv::cvtColor(img_in, labImage, cv::COLOR_BGR2Lab);

        // Split the Lab channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Perform histogram equalization on the L channel
        cv::equalizeHist(labChannels[0], labChannels[0]);

        // Merge the Lab channels back
        cv::Mat equalizedLabImage;
        cv::merge(labChannels, equalizedLabImage);

        // Convert the image back to the original color space
        cv::Mat equalizedImage;
        cv::cvtColor(equalizedLabImage, img_output_Labeq, cv::COLOR_Lab2BGR);
    }

    static void AdaptiveThresh(const cv::Mat &img_in, cv::Mat &img_bin_adaptive)
    {
        int block_size = 11; // it works in the range [40, 100] and beyond
        int C = 2.0;         // it works in the range [-40, 0]
        cv::adaptiveThreshold(img_in, img_bin_adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);
        ipa::imshow("Adaptive binarization", img_bin_adaptive, true, 2.0f);
    }

    static void TriangularThresh(const cv::Mat &img_in, cv::Mat &img_bin_triangle)
    {
        int T_triangle = ucas::getTriangleAutoThreshold(ucas::histogram(img_in));
        cv::threshold(img_in, img_bin_triangle, T_triangle, 255, cv::THRESH_BINARY);
        ipa::imshow("Triangle binarization", img_bin_triangle, true, 2.0f);
    }

    static void RegionGrowingHSV(cv::Mat &img_in, cv::Mat &segmentedImage)
    {
        cv::Mat hlsImage;
        cv::cvtColor(img_in, hlsImage, cv::COLOR_BGR2HSV);

        // Split the IHLS image into individual channels
        std::vector<cv::Mat> hlsChannels;
        cv::split(hlsImage, hlsChannels);

        // Extract the hue channel
        cv::Mat hueImage = hlsChannels[0];

        // Segment the hue image based on the specified hue range for the sign color
        cv::Mat hueBinary1;
        cv::inRange(hueImage, cv::Scalar(minHueValue), cv::Scalar(maxHueValue), hueBinary1);

        cv::Mat hueBinary2;
        cv::inRange(hueImage, cv::Scalar(150), cv::Scalar(180), hueBinary2);

        cv::Mat hueBinary;
        cv::bitwise_or(hueBinary1, hueBinary2, hueBinary);
        ipa::imshow("huebin", hueBinary, true, 0.5f);

        cv::imwrite(std::string(IMAGES_PATH) + "/hue.jpg", hueBinary);

        // Divide the binary image into 16x16 sub-regions and calculate seeds for saturation image
        cv::Mat seeds = cv::Mat::zeros(hueBinary.size(), CV_8UC1); // Initialize seeds matrix with zeros

        for (int y = 8; y < hueBinary.rows - 20; y += 16)
        {
            for (int x = 8; x < hueBinary.cols - 20; x += 16)
            {
                cv::Rect roi(x - 8, y - 8, 16, 16);
                cv::Mat subRegion = hueBinary(roi);
                if (cv::countNonZero(subRegion) > (16 * 16 / 3))
                {
                    seeds.at<uchar>(y, x) = 255;
                }
            }
        }

        cv::Mat binarySaturation;

        // Calculate the median value of the saturation channel
        double medianValue = Utils::calculateMedian(hlsChannels[1]);

        cv::threshold(hlsChannels[1], binarySaturation, 50, 255, cv::THRESH_BINARY);
        cv::imwrite(std::string(IMAGES_PATH) + "/sat.jpg", binarySaturation);

        //  AdaptiveThresh(hlsChannels[1], binarySaturation);
        ipa::imshow("prova", binarySaturation, true, 0.5f);
        cv::Mat seeds_prev;
        cv::Mat predicate = binarySaturation & hueBinary;
        int i = 0;
        do
        {
            seeds_prev = seeds.clone();

            cv::Mat candidates_img;
            cv::dilate(seeds, candidates_img,
                       cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            candidates_img = candidates_img - seeds;

            seeds += candidates_img & predicate;

            i++;
            //  ipa::imshow("Growing in progress", seeds, true, 0.5);
            // cv::waitKey(10);
        } while (cv::countNonZero(seeds - seeds_prev) && i < 20);

        segmentedImage = seeds;

        //  cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_CLOSE,
        //                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        ipa::imshow("segmented", segmentedImage, true, 0.5f);

        cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", segmentedImage);
    }
};
