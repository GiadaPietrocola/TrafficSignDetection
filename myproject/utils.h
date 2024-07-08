#pragma once
// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <numeric>

// for vscode work
#ifndef IMAGES_PATH
#define IMAGES_PATH "CassinoDataSet"
#endif

enum DET_LABEL
{
    TP,
    FP,
    FN
};

// //------------------------------------------------------------------------------
// //======================= CONSTANTS =================================

// Rectangle parameters
int min_rect_area = 800;
int max_rect_area = 12000;

float min_rectangularity = 0.75;

float aspect_ratio = 0.16;
float max_aspect_ratio_difference = 0.3;

float angle_tolerance = 45.0;
float min_horizontal_angle = 90.0 - angle_tolerance;
float max_horizontal_angle = 90.0 + angle_tolerance;

// Values of Hue to pick the red color
int minHueValue1 = 0;
int maxHueValue1 = 20;
int minHueValue2 = 160;
int maxHueValue2 = 180;

float perc_area_roi = 0.04;

// Circle parameters
int minDist = 120;
int cannyTrheshold = 100;
int accumulatorTrheshold = 18;
int minRadius = 20;
int maxRadius = 150;


std::vector<std::vector<cv::Point>> realSignContours;     // contours vector of real signs, taken from Json
std::vector<std::vector<cv::Point>> candidateSignCotours; // contours vector of candidate signs

std::vector<std::vector<float>> trueFeatures;
std::vector<std::vector<float>> falseFeatures;

std::string noEntryLabelJson = "regulatory--no-entry--g1";

std::vector<cv::Rect> ROIs;
std::vector<cv::Rect> Rects;
std::vector<cv::RotatedRect> MinRects;

std::vector<cv::Rect> candidate_roi;
std::vector<cv::Rect> candidate_rects;
std::vector<cv::RotatedRect> candidate_min_rects;

struct Utils
{

    // //------------------------------------------------------------------------------
    // //======================= METHODS =================================

    static void findRectangles(cv::Mat &img_in, std::vector<std::vector<cv::Point>> &contours)
    {

        // switch to grayscale
        cv::Mat gray;
        cv::cvtColor(img_in, gray, cv::COLOR_BGR2GRAY);

        // denoising
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0.6);

        // rectangles enhancement with grayscale morphological tophat
        cv::Mat tophat;
        cv::morphologyEx(gray, tophat, cv::MORPH_TOPHAT,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 61)));
       
        // binarization
        cv::Mat edges;
        cv::adaptiveThreshold(tophat, edges, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 71, -1);


        // morpological opening
        cv::morphologyEx(edges, edges, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

        // Find contours
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

      
        // filter by area
        contours.erase(std::remove_if(contours.begin(), contours.end(),
                                      [](const std::vector<cv::Point> &object)
                                      {
                                          double area = cv::contourArea(object);
                                          return area < min_rect_area || area > max_rect_area;
                                      }),
                       contours.end());
        
        // filter by rectangularity
        contours.erase(std::remove_if(contours.begin(), contours.end(),
                                      [](const std::vector<cv::Point> &object)
                                      {
                                          double area = cv::contourArea(object);
                                          cv::RotatedRect bounding_rect = cv::minAreaRect(object);

                                          cv::Point2f vertices[4];
                                          bounding_rect.points(vertices);

                                          float width = norm(vertices[0] - vertices[1]);
                                          float height = norm(vertices[1] - vertices[2]);

                                          float bounding_rect_area = width * height;

                                          return area / bounding_rect_area < min_rectangularity;
                                      }),
                       contours.end());
       
        // filter by aspect ratio
        contours.erase(std::remove_if(contours.begin(), contours.end(),
                                      [](const std::vector<cv::Point> &object)
                                      {
                                          cv::RotatedRect rot_rect = cv::minAreaRect(object);
                                          if (rot_rect.size.width > rot_rect.size.height)
                                          {
                                              cv::swap(rot_rect.size.width, rot_rect.size.height);
                                              rot_rect.angle += 90.f;
                                          }
                                          return std::abs(rot_rect.size.aspectRatio() - max_aspect_ratio_difference) > aspect_ratio;
                                      }),
                       contours.end());
        std::cout << contours.size();
        // filter by orientation
        contours.erase(std::remove_if(contours.begin(), contours.end(),
                                      [](const std::vector<cv::Point> &object)
                                      {
                                          cv::RotatedRect rot_rect = cv::minAreaRect(object);
                                          if (rot_rect.size.width > rot_rect.size.height)
                                          {
                                              cv::swap(rot_rect.size.width, rot_rect.size.height);
                                              rot_rect.angle += 90.f;
                                          }

                                          return !(rot_rect.angle >= min_horizontal_angle && rot_rect.angle <= max_horizontal_angle);
                                      }),
                       contours.end());

        cv::Mat output = img_in.clone();
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
        clahe->setClipLimit(2.0);                  // Adjust clip limit as needed
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



    static void RegionGrowingHSV(cv::Mat &img_in, cv::Mat &seeds, cv::Mat &segmentedImage)
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
        cv::inRange(hueImage, cv::Scalar(minHueValue1), cv::Scalar(maxHueValue1), hueBinary1);

        cv::Mat hueBinary2;
        cv::inRange(hueImage, cv::Scalar(minHueValue2), cv::Scalar(maxHueValue2), hueBinary2);

        cv::Mat hueBinary;
        cv::bitwise_or(hueBinary1, hueBinary2, hueBinary);

        cv::Mat saturationBinary;

        cv::adaptiveThreshold(hlsChannels[1], saturationBinary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 71, 2);

        cv::Mat seeds_prev;
        cv::Mat predicate = saturationBinary & hueBinary;
        int i = 0;
        do
        {
            seeds_prev = seeds.clone();

            cv::Mat candidates_img;
            cv::dilate(seeds, candidates_img,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            candidates_img = candidates_img - seeds;

            seeds += candidates_img & predicate;
            i++;
        } while (cv::countNonZero(seeds - seeds_prev) && i < 60);

        segmentedImage = seeds;
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

    static cv::Rect verticesToRect(std::vector<cv::Point> vertices)
    {

        // Trova i valori x e y minimi e massimi tra i vertici
        int minX = std::min({vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x});
        int minY = std::min({vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y});
        int maxX = std::max({vertices[0].x, vertices[1].x, vertices[2].x, vertices[3].x});
        int maxY = std::max({vertices[0].y, vertices[1].y, vertices[2].y, vertices[3].y});

        // Calcola larghezza e altezza del rettangolo
        int width = maxX - minX;
        int height = maxY - minY;

        // Crea e restituisci il rettangolo
        return cv::Rect(minX, minY, width, height);
    }

    static bool circleContainsRotatedRect(const cv::Point2f &center, float radius, const cv::RotatedRect &rotatedRect)
    {
        // Extract the vertices of the rotated rectangle
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        // Check if all four vertices of the rotated rectangle are within the circle
        for (const auto &vertex : vertices)
        {
            float distance = cv::norm(vertex - center);
            if (distance > radius + 0.5)
            {
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



    static std::vector<float> HogDescriptors(const cv::Mat& img_in) {

        cv::Mat resized;

        // Resize the image if needed
         cv::resize(img_in, resized, cv::Size(80, 80));

        cv::HOGDescriptor hog(
            cv::Size(80, 80), // winSize
            cv::Size(16, 16), // blockSize
            cv::Size(8, 8),   // blockStride
            cv::Size(8, 8),   // cellSize
            9                // nbins
        );

        std::vector<float> hogFeatures;
        hog.compute(resized, hogFeatures);
        
        // Find min and max values
        auto minmax = std::minmax_element(hogFeatures.begin(), hogFeatures.end());
        float minVal = *minmax.first;
        float maxVal = *minmax.second;

        // Normalize HOG descriptors to range [0, 1]
        std::vector<float> normalizedHogFeatures(hogFeatures.size());
        std::transform(hogFeatures.begin(), hogFeatures.end(), normalizedHogFeatures.begin(),
            [minVal, maxVal](float val) { return (val - minVal) / (maxVal - minVal); });

        return normalizedHogFeatures;
    }

    static void features(const cv::Mat &img_in, std::vector<float> &features)
    {

        cv::Mat gray;

        cv::cvtColor(img_in, gray, cv::COLOR_RGB2GRAY);

        std::vector<float> hogFeatures = HogDescriptors(gray);

        features.insert(features.end(), hogFeatures.begin(), hogFeatures.end());
   
    }

    static void writeCsv(const std::vector<std::vector<float>> &feature, std::string filename)
    {
        // Save features to a CSV file
        std::ofstream file(filename, std::ios::app);
        if (file.is_open())
        {

            for (int i = 0; i < feature.size(); i++)
            {
                for (int j = 0; j < feature[i].size(); j++)
                {
                    file << feature[i][j] << ",";
                }
                file << std::endl;
            }

            file.close();
        }
        else
        {
            std::cerr << "Unable to open file to save features." << std::endl;
            return;
        }
    }
    
    static void ShowMachineLearningResults(cv::Mat& img_in, const std::string filename, int index, cv::Rect rectangle,cv::Scalar color )
    {

        std::ifstream file(filename);
        std::string line;

        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        // Skip the header line
        std::getline(file, line);

        // Read the rest of the lines
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            int sampleIndex;
            float score;
            ss >> sampleIndex >> score;
            if (index== sampleIndex && score > 0.25) {
                cv::rectangle(img_in, rectangle, color, 2);
                cv::putText(img_in, std::to_string(score).substr(0, 6), cv::Point(rectangle.x+rectangle.width+10, rectangle.y + rectangle.height/2), cv::FONT_HERSHEY_SIMPLEX, 1.5, color, 3);

            }
        }
    }
};
