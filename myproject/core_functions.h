// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"

// include my project functions
#include "functions.h"

#include <iostream>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

using namespace rapidjson;

struct CoreFunctions
{
    /**
     * @brief Main pipeline function for image processing.
     *
     * @param img_in Input image to be processed.
     * @param show Boolean flag to display images at various stages.
     */
    static void Pipeline(cv::Mat &img_in, bool show)
    {
        cv::Mat img_copy = img_in.clone();

        cv::Mat preProcessedImg = Preprocessing(img_in);

        std::vector<std::vector<cv::Point>> contours;

        Utils::findRectangles(preProcessedImg, contours);

        // //------------------------------------------------------------------------------
        // //======================= ROI CREATION =================================
        // //------------------------------------------------------------------------------
        ROIs.clear();
        Rects.clear();
        MinRects.clear();

        PipeRoiCreation(preProcessedImg, contours);

        // //------------------------------------------------------------------------------
        // //======================= REGION GROWING =================================
        // //------------------------------------------------------------------------------
        candidate_roi.clear();
        candidate_rects.clear();
        candidate_min_rects.clear();

        PipeRegionGrowing(preProcessedImg);

        // //------------------------------------------------------------------------------
        // //======================= HOUGH CIRCLES =================================
        // //------------------------------------------------------------------------------
        PipeHoughCircles(preProcessedImg,img_in);

        // ipa::imshow("img", img_copy, true);
        // ipa::imshow("img", img_in, true);
    }

    /**
     * @brief Preprocess the input image.
     *
     * @param img_in Input image to be preprocessed.
     * @return cv::Mat Preprocessed image.
     */
    static cv::Mat Preprocessing(cv::Mat &img_in)
    {
        cv::Mat img_eq;
        Utils::HistogramLabeq(img_in, img_eq);

        return img_eq;
    }

    /**
     * @brief Create ROIs from contours.
     *
     * @param img_in Input image.
     * @param contours Vector of contours detected in the image.
     */
    static void PipeRoiCreation(cv::Mat &img_in, std::vector<std::vector<cv::Point>> contours)
    {
        for (const auto &contour : contours)
        {

            // Get the bounding rectangle enclosing the contour
            cv::Rect bounding_rect = cv::boundingRect(contour);
            cv::RotatedRect min_bounding_rect = cv::minAreaRect(contour);

            // Calculate the width and height of the bounding rectangle
            int width = bounding_rect.width;
            int height = bounding_rect.height;

            // Center of the rectangle
            cv::Point center(bounding_rect.x + width / 2, bounding_rect.y + height / 2);

            // Define the ROI
            int roi_width = 2.5 * width;
            int roi_height = 2.5 * width;

            // Adjust ROI to ensure it remains within image bounds
            int roi_x = std::max(0, center.x - roi_width / 2);
            int roi_y = std::max(0, center.y - roi_height / 2);
            int roi_right = std::min(img_in.cols - 1, roi_x + roi_width);
            int roi_bottom = std::min(img_in.rows - 1, roi_y + roi_height);

            // Create ROI rect
            cv::Rect ROI(roi_x, roi_y, roi_right - roi_x, roi_bottom - roi_y);

            // Add the ROI to the list
            ROIs.push_back(ROI);
            Rects.push_back(bounding_rect);
            MinRects.push_back(min_bounding_rect);
        }
    }

    /**
     * @brief Perform region growing on the preprocessed image.
     *
     * @param preProcessedImg Preprocessed input image.
     */
    static void PipeRegionGrowing(cv::Mat &preProcessedImg)
    {
        for (size_t k = 0; k < ROIs.size(); k++)
        {

            cv::Mat img_roi = preProcessedImg(ROIs[k]).clone();
            cv::Mat roi_gray;
            cv::cvtColor(img_roi, roi_gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(roi_gray, roi_gray, cv::Size(3, 3), 0.5);

            cv::Mat seeds = cv::Mat::zeros(img_roi.size(), CV_8UC1); // Initialize seeds matrix with zeros

            // Assuming Rects[k] represents your rectangle
            int centerX = Rects[k].x - ROIs[k].x + Rects[k].width / 2;  // X coordinate of the center of the rectangle
            int centerY = Rects[k].y - ROIs[k].y + Rects[k].height / 2; // Y coordinate of the center of the rectangle

            // Check if adding seeds won't exceed image boundaries
            if (centerY - 4 - Rects[k].height / 2 >= 0 && centerY - 4 - Rects[k].height / 2 < img_roi.rows)
                seeds.at<uchar>(centerY - 4 - Rects[k].height / 2, centerX) = 255; // Over the center

            if (centerY + 4 + Rects[k].height / 2 >= 0 && centerY + 4 + Rects[k].height / 2 < img_roi.rows)
                seeds.at<uchar>(centerY + 4 + Rects[k].height / 2, centerX) = 255; // Under the center

            if (centerX - 4 - Rects[k].width / 2 >= 0 && centerX - 8 - Rects[k].width / 2 < img_roi.cols)
                seeds.at<uchar>(centerY, centerX - 4 - Rects[k].width / 2) = 255; // Left of the center

            if (centerX + 8 + Rects[k].width / 2 >= 0 && centerX + 8 + Rects[k].width / 2 < img_roi.cols)
                seeds.at<uchar>(centerY, centerX + 4 + +Rects[k].width / 2) = 255; // Right of the center

            //  ipa::imshow("seeds", seeds, true);

            cv::Mat segmentedImage;

            Utils::RegionGrowingHSV(img_roi, seeds, segmentedImage);

            int count = 0;
            for (int i = 0; i < img_roi.rows; i++)
            {
                for (int j = 0; j < img_roi.cols; j++)
                {
                    if (segmentedImage.at<uchar>(i, j) == 255)
                        count++;
                }
            }
            //   std::cout << count<<"\n";
            //   std::cout << img_roi.rows * img_roi.cols << "\n";

            //   ipa::imshow("seg", segmentedImage, true);
            if (count > 0.04 * img_roi.rows * img_roi.cols)
            {
                candidate_roi.push_back(ROIs[k]);
                candidate_rects.push_back(Rects[k]);
                candidate_min_rects.push_back(MinRects[k]);
            }
        }
    }

    /**
     * @brief Detect circles using Hough Transform.
     *
     * @param preProcessedImg Preprocessed input image.
     */
    static void PipeHoughCircles(cv::Mat &preProcessedImg, cv::Mat img_in)
    {
        cv::Mat tmp = preProcessedImg.clone();
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);

        int c = 0;

        for (size_t k = 0; k < candidate_roi.size(); k++)
        {

            cv::Mat img_roi = preProcessedImg(candidate_roi[k]).clone();
            cv::Mat roi_gray;
            cv::cvtColor(img_roi, roi_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Vec3f> circles;
            cv::GaussianBlur(roi_gray, roi_gray, cv::Size(3, 3), 0.5);

            HoughCircles(roi_gray, circles, cv::HOUGH_GRADIENT, 2, 120, 100, 18, 20, 150);

            if (!circles.empty())
            {

                std::vector<cv::Rect> candidate;

                // Draw the circles on the original image
                for (size_t i = 0; i < circles.size(); i++)
                {

                    cv::Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);

                    center.x += candidate_roi[k].x; // Adjust for ROI offset
                    center.y += candidate_roi[k].y; // Adjust for ROI offset
                    if (Utils::circleContainsRotatedRect(center, radius, candidate_min_rects[k]))
                    {

                        center.x -= ROIs[k].x; // Adjust for ROI offset
                        center.y -= ROIs[k].y; // Adjust for ROI offset

                        circle(img_roi, center, radius, cv::Scalar(0, 255, 0), 4);
                        circle(img_roi, center, 5, cv::Scalar(0, 128, 255), -1);

                        cv::Rect boundingRect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
                        boundingRect.x += candidate_roi[k].x; // Adjust for ROI offset
                        boundingRect.y += candidate_roi[k].y; // Adjust for ROI offset
                        candidate.push_back(boundingRect);

                        //   cv::rectangle(img_in, boundingRect, cv::Scalar(255, 0, 0), 2);

                        //   ipa::imshow("roi", img_roi, true);
                    }
                }

                if (candidate.size() > 0)
                {

                    // Define the ROI
                    int rect_width = 1.5 * candidate_rects[k].width;
                    int rect_height = 1.5 * candidate_rects[k].width;

                    // Center of the rectangle
                    cv::Point center(candidate_rects[k].x + candidate_rects[k].width / 2, candidate_rects[k].y + candidate_rects[k].height / 2);

                    // Adjust rect to ensure it remains within image bounds
                    int rect_x = std::max(0, center.x - rect_width / 2);
                    int rect_y = std::max(0, center.y - rect_height / 2);
                    int rect_right = std::min(preProcessedImg.cols - 1, rect_x + rect_width);
                    int rect_bottom = std::min(preProcessedImg.rows - 1, rect_y + rect_height);

                    // Create rect
                    cv::Rect bounding(rect_x, rect_y, rect_right - rect_x, rect_bottom - rect_y);

                    candidateSignCotours.push_back(Utils::getRectContours(bounding));
                    
                    cv::rectangle(tmp, bounding, cv::Scalar(255, 0, 0), 2);
                    cv::Mat roi = preProcessedImg(bounding).clone();
                    // ipa::imshow("roi", roi, true);

                    std::vector<float> roiFeatures;
                    Utils::features(roi, roiFeatures);
                    // Track the best intersection for this ROI
                    float bestIntersection = 0.0;
                    int bestContourIndex = -1;

                    cv::Mat img_copy = img_in.clone();

                    for (int j = 0; j < realSignContours.size(); j++) {
                        float intersection = Utils::IntersectionOverUnion(Utils::verticesToRect(realSignContours[j]), bounding);

                        if (intersection > bestIntersection) {
                            bestIntersection = intersection;
                            bestContourIndex = j;
                        }
                    }

                    // If the best intersection is above the threshold, it's a true positive
                    if (bestIntersection > 0.5) {
                        trueFeatures.push_back(roiFeatures);
                        Utils::ShowMachineLearningResults(img_copy, "10-fold-pos.SEL(001)15.sco", trueFeatures.size(), bounding, cv::Scalar(0, 255, 0));
                        // Remove or mark the ground truth contour as matched
                        realSignContours.erase(realSignContours.begin() + bestContourIndex);
                    }
                    else if (c < 1) {
                        falseFeatures.push_back(roiFeatures);
                        Utils::ShowMachineLearningResults(img_copy, "10-fold-neg.SEL(001)15.sco", falseFeatures.size(), bounding, cv::Scalar(0, 0, 255));
                        c++;
                    }

                }

                
            }
        }
       // ipa::imshow("Hope", tmp, true);
    }

    /**
     * @brief Function that handles all the Json stuff.
     *
     * @param show If true, shows the images with detected signs.
     */
    static void JsonHandler(bool show)
    {
        const int total_number = 96;
        int ok = 0;

        for (int k = 51; k <= total_number; k++)
        {
            realSignContours.clear();
            candidateSignCotours.clear();

            std::string path = std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".jpg";
            std::string path_json = std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".json";

            Document document;
            if (!LoadJsonDocument(path_json, document))
                continue;

            cv::Mat img = cv::imread(path);
            cv::Mat img2draw = img.clone();

            ExtractRealSignContours(document, img2draw);
            if (show)
            {
                ipa::imshow("Real Sign", img2draw, true, 0.5f);
                ipa::imshow("Original image", img, true, 0.5f);
            }

            CoreFunctions::Pipeline(img, show);
            float max_intersection = CalculateMaxIntersection();

            LogDetectionResult(k, max_intersection, ok);
            cv::destroyAllWindows();
        }

        Utils::writeCsv(falseFeatures, "false_glcm_features_test.csv");
        Utils::writeCsv(trueFeatures, "true_glcm_features_test.csv");

        printf("Number of ok: %d on %d\n", ok, total_number);
        printf("Percentual %f", ((float)ok / total_number) * 100);
    }

    /**
     * @brief Loads a JSON document from the specified file path.
     *
     * @param path_json The path to the JSON file.
     * @param document The JSON document to load into.
     * @return True if the JSON document was successfully loaded, false otherwise.
     */
    static bool LoadJsonDocument(const std::string &path_json, Document &document)
    {
        FILE *file = fopen(path_json.c_str(), "rb");
        if (!file)
        {
            printf("Error opening file: %s\n", path_json.c_str());
            return false;
        }

        char buffer[65536];
        FileReadStream inputStream(file, buffer, sizeof(buffer));
        document.ParseStream(inputStream);
        fclose(file);

        if (document.HasParseError())
        {
            printf("Error parsing JSON: %s\n", path_json.c_str());
            return false;
        }
        return true;
    }

    /**
     * @brief Extracts the real sign contours from the JSON document and draws them on the image.
     *
     * @param document The JSON document containing the sign information.
     * @param img2draw The image on which to draw the contours.
     */
    static void ExtractRealSignContours(Document &document, cv::Mat &img2draw)
    {
        for (auto &obj : document["objects"].GetArray())
        {
            if (strcmp(obj["label"].GetString(), noEntryLabelJson.c_str()) == 0)
            {
                std::vector<cv::Point> bbox;
                rapidjson::Value bboxvalue = obj["bbox"].GetArray();
                bbox.push_back(cv::Point(bboxvalue["xmin"].GetFloat(), bboxvalue["ymin"].GetFloat()));
                bbox.push_back(cv::Point(bboxvalue["xmax"].GetFloat(), bboxvalue["ymin"].GetFloat()));
                bbox.push_back(cv::Point(bboxvalue["xmax"].GetFloat(), bboxvalue["ymax"].GetFloat()));
                bbox.push_back(cv::Point(bboxvalue["xmin"].GetFloat(), bboxvalue["ymax"].GetFloat()));
                realSignContours.push_back(bbox);

                cv::drawContours(img2draw, realSignContours, -1, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    /**
     * @brief Calculates the maximum intersection-over-union (IoU) between real sign contours and candidate sign contours.
     *
     * @return The maximum IoU value.
     */
    static float CalculateMaxIntersection()
    {
        float max_intersection = 0;
        for (const auto &realSignContour : realSignContours)
        {
            for (const auto &candidateSignContour : candidateSignCotours)
            {
                float intersection = Utils::IntersectionOverUnion(
                    Utils::verticesToRect(realSignContour),
                    Utils::verticesToRect(candidateSignContour));
                if (intersection > max_intersection)
                    max_intersection = intersection;
            }
        }
        return max_intersection;
    }

    /**
     * @brief Logs the detection result for a given sign.
     *
     * @param k The sign index.
     * @param max_intersection The maximum intersection-over-union (IoU) value.
     * @param ok The count of correctly detected signs.
     */
    static void LogDetectionResult(int k, float max_intersection, int &ok)
    {
        std::string msg = "Sign" + std::to_string(k) + (max_intersection > 0.5 ? ": ok" : ": no");
        printf("%s\n", msg.c_str());

        if (max_intersection > 0.5)
            ok++;
    }
};