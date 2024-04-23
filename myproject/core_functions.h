// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"

// include my project functions
#include "functions.h"

// include my project functions
#ifndef IMAGES_PATH
#define IMAGES_PATH "example_images"
#endif

struct CoreFunctions
{
    // Function to calculate the median of a single-channel image
    static double calculateMedian(const cv::Mat& channel) {
        // Create a vector to store pixel values
        std::vector<uchar> pixels;
        pixels.reserve(channel.rows * channel.cols);

        // Copy pixel values to the vector
        for (int y = 0; y < channel.rows; ++y) {
            for (int x = 0; x < channel.cols; ++x) {
                pixels.push_back(channel.at<uchar>(y, x));
            }
        }

        // Sort the pixel values
        std::sort(pixels.begin(), pixels.end());

        // Calculate the median
        if (pixels.size() % 2 == 0) {
            // If the number of pixels is even, take the average of the two middle values
            return static_cast<double>(pixels[pixels.size() / 2 - 1] + pixels[pixels.size() / 2]) / 2.0;
        }
        else {
            // If the number of pixels is odd, take the middle value
            return static_cast<double>(pixels[pixels.size() / 2]);
        }
    }

    static void Preprocessing(cv::Mat &img_in)
    {
        float img_area = img_in.rows * img_in.cols;
        cv::Mat blurred;
        cv::Mat edges;


       
        Utils::RealCanny(img_in, img_in, edges, sigma);

        cv::Mat hlsImage;
        cv::cvtColor(img_in , hlsImage, cv::COLOR_BGR2HLS);

        // Split the IHLS image into individual channels
        std::vector<cv::Mat> hlsChannels;
        cv::split(hlsImage, hlsChannels);

        // Extract the hue channel
        cv::Mat hueImage = hlsChannels[0];

        int minHueValue = 0;
        int maxHueValue = 50;
        // Segment the hue image based on the specified hue range for the sign color
        cv::Mat hueBinary1;
        cv::inRange(hueImage, cv::Scalar(minHueValue), cv::Scalar(maxHueValue), hueBinary1);

        cv::Mat hueBinary2;
        cv::inRange(hueImage, cv::Scalar(150), cv::Scalar(180), hueBinary2);

        cv::Mat hueBinary;
        cv::bitwise_or(hueBinary1, hueBinary2, hueBinary);
        ipa::imshow("huebin", hueBinary, true, 0.5f);


        // Divide the binary image into 16x16 sub-regions and calculate seeds for saturation image
        cv::Mat seeds = cv::Mat::zeros(hueBinary.size(), CV_8UC1); // Initialize seeds matrix with zeros
        
        for (int y = 8; y < hueBinary.rows-20; y += 16) {
            for (int x = 8; x < hueBinary.cols-20; x += 16) {
                cv::Rect roi(x - 8, y - 8, 16, 16);
                cv::Mat subRegion = hueBinary(roi);
                if (cv::countNonZero(subRegion) > (16 * 16 / 3)) {
                    seeds.at<uchar>(y, x) = 255;
                }
            }
        }

        cv::Mat binarySaturation;
        // Calculate the median value of the saturation channel
        double medianValue = calculateMedian(hlsChannels[2]);
        cv::threshold(hlsChannels[2], binarySaturation, medianValue, 255, cv::THRESH_BINARY);

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
            //ipa::imshow("Growing in progress", seeds, false, 0.5);
            //cv::waitKey(10);
        } while (cv::countNonZero(seeds - seeds_prev)&& i<50);

        cv::Mat segmentedImage = seeds;

        cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));



        ipa::imshow("segmented", segmentedImage, true, 0.5f);

        cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", segmentedImage);

     //   Utils::RealCanny(img_in, blurred, edges, sigma);

      //  ipa::imshow("Edges", edges, true, 0.5f);

        // extract objects
        std::vector<std::vector<cv::Point>> objects;
        cv::findContours(segmentedImage, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // specialized parameters
        int min_width = min_width_perc * img_in.cols + 0.5f;
        int max_width = max_width_perc * img_in.cols + 0.5f;

        int min_area = min_area_perc * img_area + 0.5f;
        int max_area = max_area_perc * img_area + 0.5f;

        // discard objects that are not in the [min_area, max_area]
        std::vector<std::vector<cv::Point>> candidate_objects;
        for (int k = 0; k < objects.size(); k++)
        {
            float area = cv::contourArea(objects[k]);
            cv::Rect brect = cv::boundingRect(objects[k]);
            if (area >= min_area && area <= max_area)
                candidate_objects.push_back(objects[k]);
        }

        // discard objects that are below the minimun circularity
        std::vector<std::vector<cv::Point>> circular_objects;
        for (int k = 0; k < candidate_objects.size(); k++)
        {
            float area = cv::contourArea(candidate_objects[k]);
            float perim = cv::arcLength(candidate_objects[k], true);
            float circularity = 4 * ucas::PI * area / (perim * perim);
            if (circularity >= min_circularity)
                circular_objects.push_back(candidate_objects[k]);
        }

        cv::drawContours(img_in, circular_objects, -1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

        ipa::imshow("image", img_in, true, 0.5f);

        std::vector < cv::Mat> roi;
        for (int k = 0; k < circular_objects.size(); k++)
        {
            cv::Rect boundingRect = cv::boundingRect(circular_objects[k]);
            roi.push_back(img_in(boundingRect).clone());

            ipa::imshow("ROI", roi[k], true);
        }
        
        // Applica il filtro di Canny per rilevare i bordi

        cv::Mat gray;
        cv::Mat tmp;
        std::vector < cv::Mat> roie;

        for (int k = 0; k < circular_objects.size(); k++)
        {
            Utils::RealCanny(roi[k], gray,tmp, sigma);
            roie.push_back(tmp);
            ipa::imshow("ROI", roie[k], true);

            // extract objects
            std::vector<std::vector<cv::Point>> robjects;
            cv::findContours(roie[k], robjects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

            // Calcola il bounding box per ciascun contorno e la differenza tra l'area del bounding box e l'area del contorno
            for (size_t i = 0; i < robjects.size(); ++i)
            {
                // Calcola il bounding box del contorno
                cv::Rect boundingRect = cv::boundingRect(robjects[i]);

                if (boundingRect.width >= 20 && boundingRect.width <= 100)
                {

                    // Calcola l'area del bounding box e l'area del contorno
                    double boundingArea = boundingRect.width * boundingRect.height;
                    double contourArea = cv::contourArea(robjects[i]);

                    // Calcola la differenza tra le aree
                    double areaDifference = boundingArea - contourArea;

                    std::cout << "Contour " << i << ": Bounding area = " << boundingArea
                        << ", Contour area = " << contourArea
                        << ", Area difference = " << areaDifference << std::endl;

                    if (areaDifference < 0.3 * boundingArea)
                    {
                        // Disegna il bounding box sulle immagine originale (solo per debugging)
                        cv::rectangle(roi[k], boundingRect, cv::Scalar(255), 1);
                    }
                }
            }

            // Visualizza l'immagine con i bounding box disegnati
            ipa::imshow("Bounding Boxes", roi[k], true);
        }
        

       


    }
};