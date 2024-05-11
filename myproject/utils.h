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
float min_circularity = 0.6f; // filter parameter: minimum circularity normalized
float sigma = 1.0f;
float min_width_perc = 0.08f; // filter parameter: minimum width of light expressed as percentage of image width
float max_width_perc = 0.8f;  // filter parameter: maximum width of light expressed as percentage of image width

float min_area_perc = 0.0002f; // filter parameter: minimum area of signs extressed as percentage of image area
float max_area_perc = 0.5f;    // filter parameter: maximum area of signs extressed as percentage of image area

int minHueValue = 0;  // filter parameter: minimum value of Hue to pick the red color
int maxHueValue = 20; // filter parameter: maximum value of Hue to pick the red color

std::vector<std::vector<cv::Point>> realSignContours;     // contours vector of real signs, taken from Json
std::vector<std::vector<cv::Point>> candidateSignCotours; // contours vector of candidate signs

std::string noEntryLabelJson = "regulatory--no-entry--g1";

struct Utils
{
    
    // //------------------------------------------------------------------------------
    // //======================= METHODS =================================
 

    static void findRectangles(cv::Mat& img_in, std::vector< std::vector<cv::Point> > &contours) {

        // switch to grayscale
        cv::Mat gray;
        cv::cvtColor(img_in, gray, cv::COLOR_BGR2GRAY);

        // denoising
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0,0.6);

        // rectangles enhancement with grayscale morphological tophat
        cv::Mat tophat;
        cv::morphologyEx(gray, tophat, cv::MORPH_TOPHAT,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 61)));

        cv::Mat edges;
      
        // binarization
        Utils::AdaptiveThresh(tophat, edges,71,-1);
        
        // morpological opening
        cv::morphologyEx(edges, edges, cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

        // Find contours    
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // filter by area
        contours.erase(std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& object)
            {
                double area = cv::contourArea(object);
                return area<800 ||area > 12000;
            }
        ), contours.end());
        
        // filter by rectangularity
        contours.erase(std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& object)
            {
                double area = cv::contourArea(object);
                cv::RotatedRect bounding_rect = cv::minAreaRect(object);

                cv::Point2f vertices[4];
                bounding_rect.points(vertices);

                float width = norm(vertices[0] - vertices[1]);
                float height = norm(vertices[1] - vertices[2]);

                float bounding_rect_area = width * height;

                return area / bounding_rect_area < 0.75;
            }
        ), contours.end());

        // filter by aspect ratio
        contours.erase(std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& object)
            {
                cv::RotatedRect rot_rect = cv::minAreaRect(object);
                if (rot_rect.size.width > rot_rect.size.height)
                {
                    cv::swap(rot_rect.size.width, rot_rect.size.height);
                    rot_rect.angle += 90.f;
                }
                return std::abs(rot_rect.size.aspectRatio() - 0.3) > 0.16;
            }
        ), contours.end());
        
        // filter by orientation
        contours.erase(std::remove_if(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& object)
            {
                cv::RotatedRect rot_rect = cv::minAreaRect(object);
                if (rot_rect.size.width > rot_rect.size.height)
                {
                    cv::swap(rot_rect.size.width, rot_rect.size.height);
                    rot_rect.angle += 90.f;
                }
               
        float angle_tolerance = 45.0;
        float min_horizontal_angle = 90.0 - angle_tolerance;
        float max_horizontal_angle = 90.0 + angle_tolerance;

        return !(rot_rect.angle >= min_horizontal_angle && rot_rect.angle <= max_horizontal_angle);

            }
        ), contours.end());
       
        cv::Mat output = img_in.clone();
    }


    static float Median(const cv::Mat &img_in) {
        // Convertiamo la matrice in un vettore
        std::vector<int> values;
        if (img_in.isContinuous()) {
            values.assign(img_in.data, img_in.data + img_in.total());
        }
        else {
            for (int i = 0; i < img_in.rows; ++i) {
                values.insert(values.end(), img_in.ptr<int>(i), img_in.ptr<int>(i) + img_in.cols);
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
        return median;
    }
    /**
     * @brief Real version of Canny's Algorithm
     *
     * @param img_in Input image
     * @param img_blurred Output image after Gaussian Blur
     * @param edges Output edge's image after Canny's algorithm
     * @param sigma Sigma coefficient of Gaussian Blur
     */
    static void RealCanny(cv::Mat &img_in, cv::Mat &img_blurred, cv::Mat &edges, int sigma)
    {
      //  cv::cvtColor(img_in, img_in, cv::COLOR_BGR2GRAY);

     //   Utils::CustomGaussianBlur(img_in, img_blurred, sigma);


        cv::GaussianBlur(img_in, img_blurred, cv::Size(5, 5), 0.5, 0.5);
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

       // std::cout << median;
        int lower = int(std::max(0, int(0.9*median)));
        int upper = int(std::min(255, int(1.2 * median)));


        cv::Canny(img_blurred, edges, 150/2, 150, 3, false);

        // A Close operation could be useful??
        //  cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
        //                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

         cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", edges);
    }


    /**
     * @brief Histogram Equalization on LAB channels
     *
     * @param img_in Input Image
     * @param img_output_Labeq Output Image
     */
    static void HistogramLabeq(const cv::Mat &img_in, cv::Mat &img_output_Labeq)
    {
        // Convert the image to Lab color space
        cv::Mat labImage;
        cv::cvtColor(img_in, labImage, cv::COLOR_BGR2Lab);

        // Split the Lab channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);

        // Perform histogram equalization on the L channel
        // Create CLAHE object
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2.0); // Adjust clip limit as needed
        clahe->setTilesGridSize(cv::Size(10, 10)); // Adjust grid size as needed

        // Apply CLAHE
        cv::Mat equalized;
        clahe->apply(labChannels[0], labChannels[0]);


        // Merge the Lab channels back
        cv::Mat equalizedLabImage;
        cv::merge(labChannels, equalizedLabImage);

        // Convert the image back to the original color space
        cv::Mat equalizedImage;
        cv::cvtColor(equalizedLabImage, img_output_Labeq, cv::COLOR_Lab2BGR);
    }

    /**
     * @brief Adaptive Threshold
     *
     * @param img_in Input Image
     * @param img_bin_adaptive Output Image
     */
    static void AdaptiveThresh(const cv::Mat &img_in, cv::Mat &img_bin_adaptive,int block_size, int c)
    {
  
        cv::adaptiveThreshold(img_in, img_bin_adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, c);
        //ipa::imshow("Adaptive binarization", img_bin_adaptive, true, 2.0f);
    }
    
    static void RegionGrowingHSV(cv::Mat & img_in, cv::Mat& seeds, cv::Mat & segmentedImage)
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
        cv::inRange(hueImage, cv::Scalar(160), cv::Scalar(180), hueBinary2);

        cv::Mat hueBinary;
        cv::bitwise_or(hueBinary1, hueBinary2, hueBinary);

    //    ipa::imshow("huebin", hueBinary, true, 0.5f);

        cv::imwrite(std::string(IMAGES_PATH) + "/hue.jpg", hueBinary);

      
        cv::Mat binarySaturation;

        // Calculate the median value of the saturation channel
        double medianValue = Utils::Median(hlsChannels[1]);

        //  cv::threshold(hlsChannels[1], binarySaturation, 20, 255, cv::THRESH_BINARY);
          //cv::imwrite(std::string(IMAGES_PATH) + "/sat.jpg", binarySaturation);

        AdaptiveThresh(hlsChannels[1], binarySaturation,71,2);
      //  ipa::imshow("prova", binarySaturation, true, 0.5f);
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
        } while (cv::countNonZero(seeds - seeds_prev) && i<60);

        segmentedImage = seeds;


        cv::imwrite(std::string(IMAGES_PATH) + "/edges.jpeg", segmentedImage);
    }


    /**
     * @brief Intersection Over Union
     *
     * @param rect1 First Input Rect
     * @param rect2 Second Input Rect
     * @return float Intersection Over Union
     */
    static float IntersectionOverUnion(cv::Rect rect1, cv::Rect rect2)
    {
        float areaOfOverlap = 0;
        float areaOfUnion = 0;

        cv::Rect intersection_rect = rect1 & rect2;

        if (!intersection_rect.empty())
            areaOfOverlap = intersection_rect.width * intersection_rect.height;
        else
            return -1;

        areaOfUnion = rect1.area() + rect2.area() - areaOfOverlap;

        return (areaOfOverlap / areaOfUnion);
    }
   
    static cv::Rect verticesToRect(std::vector<cv::Point> vertices) {

        // Trova i valori x e y minimi e massimi tra i vertici
        int minX = std::min({ vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x });
        int minY = std::min({ vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y });
        int maxX = std::max({ vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x });
        int maxY = std::max({ vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y });

        // Calcola larghezza e altezza del rettangolo
        int width = maxX - minX;
        int height = maxY - minY;

        // Crea e restituisci il rettangolo
        return cv::Rect(minX, minY, width, height);
    }

    static bool circleContainsRotatedRect(const cv::Point2f& center, float radius, const cv::RotatedRect& rotatedRect) {
        // Extract the vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        // Check if all four vertices of the rotated rectangle are within the circle
        for (const auto& vertex : vertices) {
            float distance = cv::norm(vertex - center);
            if (distance > radius+0.5) {
                return false; // If any vertex is outside the circle, return false
            }
        }

        return true; // If all vertices are within the circle, return true
    }

    static std::vector<cv::Point> getRectContours(cv::Rect rect)
    {
        std::vector<cv::Point> tmp;
        tmp.push_back(cv::Point(rect.x, rect.y));
        tmp.push_back(cv::Point(rect.x + rect.width, rect.y));
        tmp.push_back(cv::Point(rect.x + rect.width, rect.y + rect.height));
        tmp.push_back(cv::Point(rect.x, rect.y + rect.height));
        return tmp;
    }

};
