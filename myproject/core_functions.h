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

    static void Preprocessing(cv::Mat &img_in, bool show)
    {
       
       std::vector< std::vector<cv::Point> > contours;

       cv::Mat img_eq;
       Utils::HistogramLabeq(img_in, img_eq);
  
       Utils::findRectangles(img_eq, contours);
       
       std::vector<cv::Rect> ROIs;
       std::vector<cv::Rect> Rects;
       std::vector<cv::RotatedRect> MinRects;
     
       for (const auto& contour : contours) {

          // Get the bounding rectangle enclosing the contour
          cv::Rect bounding_rect = cv::boundingRect(contour);
          cv::RotatedRect min_bounding_rect = cv::minAreaRect(contour);

          // Calculate the width and height of the bounding rectangle
          int width = bounding_rect.width;
          int height = bounding_rect.height;

          // Center of the rectangle
          cv::Point center(bounding_rect.x + width / 2, bounding_rect.y + height / 2);

          // Define the ROI
          int roi_width = 2.5* width;
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

      std::vector<cv::Rect> candidate_roi;
      std::vector<cv::Rect> candidate_rects;
      std::vector<cv::RotatedRect> candidate_min_rects;

      for (size_t k = 0; k < ROIs.size(); k++) {

          cv::Mat img_roi = img_eq(ROIs[k]).clone();
          cv::Mat roi_gray;
          cv::cvtColor(img_roi, roi_gray, cv::COLOR_BGR2GRAY);
          std::vector<cv::Vec3f> circles;
          cv::GaussianBlur(roi_gray, roi_gray, cv::Size(3, 3), 0.5);

          cv::Mat seeds = cv::Mat::zeros(img_roi.size(), CV_8UC1); // Initialize seeds matrix with zeros

          // Assuming Rects[k] represents your rectangle
          int centerX = Rects[k].x - ROIs[k].x + Rects[k].width / 2; // X coordinate of the center of the rectangle
          int centerY = Rects[k].y - ROIs[k].y + Rects[k].height / 2; // Y coordinate of the center of the rectangle

          // Check if adding seeds won't exceed image boundaries
          if (centerY - 4 - Rects[k].height / 2 >= 0 && centerY - 4 - Rects[k].height / 2 < img_roi.rows) {
              seeds.at<uchar>(centerY - 4 - Rects[k].height / 2, centerX) = 255; // Over the center

          }

          if (centerY + 4 + Rects[k].height / 2 >= 0 && centerY + 4 + Rects[k].height / 2 < img_roi.rows) {
              seeds.at<uchar>(centerY + 4 + Rects[k].height / 2, centerX) = 255; // Under the center

          }

          if (centerX - 4 - Rects[k].width / 2 >= 0 && centerX - 8 - Rects[k].width / 2 < img_roi.cols) {
              seeds.at<uchar>(centerY, centerX - 4 - Rects[k].width / 2) = 255; // Left of the center
          }

          if (centerX + 8 + Rects[k].width / 2 >= 0 && centerX + 8 + Rects[k].width / 2 < img_roi.cols) {
              seeds.at<uchar>(centerY, centerX + 4 + +Rects[k].width / 2) = 255; // Right of the center
          }

          //  ipa::imshow("seeds", seeds, true);

          cv::Mat segmentedImage;

          Utils::RegionGrowingHSV(img_roi, seeds, segmentedImage);

          //   cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_CLOSE,
            //                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));


          int count = 0;
          for (int i = 0; i < img_roi.rows; i++) {
              for (int j = 0; j < img_roi.cols; j++) {
                  if (segmentedImage.at<uchar>(i, j) == 255) {
                      count++;
                  }
              }
          }
          //   std::cout << count<<"\n";
          //   std::cout << img_roi.rows * img_roi.cols << "\n";

          //   ipa::imshow("seg", segmentedImage, true);
          if (count > 0.04 * img_roi.rows * img_roi.cols) {
              candidate_roi.push_back(ROIs[k]);
              candidate_rects.push_back(Rects[k]);
              candidate_min_rects.push_back(MinRects[k]);
          }
      }
 

    //hough

      for (size_t k = 0; k < candidate_roi.size(); k++) {

          cv::Mat img_roi = img_eq(candidate_roi[k]).clone();

       //   ipa::imshow("a", img_roi, true);
          cv::Mat roi_gray;
          cv::cvtColor(img_roi, roi_gray, cv::COLOR_BGR2GRAY);
          std::vector<cv::Vec3f> circles;
          cv::GaussianBlur(roi_gray, roi_gray, cv::Size(3, 3), 0.5);

          HoughCircles(roi_gray, circles, cv::HOUGH_GRADIENT, 2, 120, 100, 18, 20, 150);

          if (!circles.empty()) {

              std::vector<cv::Rect> candidate;
              
              // Draw the circles on the original image
              for (size_t i = 0; i < circles.size(); i++) {

                  cv::Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                  int radius = cvRound(circles[i][2]);

                  center.x += candidate_roi[k].x; // Adjust for ROI offset
                  center.y += candidate_roi[k].y; // Adjust for ROI offset
                 if (Utils::circleContainsRotatedRect(center, radius, candidate_min_rects[k])) {

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

              
              if(candidate.size()>0) {

                  // Define the ROI
                  int rect_width = 1.5 * candidate_rects[k].width;
                  int rect_height = 1.5 * candidate_rects[k].width;

                  // Center of the rectangle
                  cv::Point center(candidate_rects[k].x + candidate_rects[k].width / 2, candidate_rects[k].y + candidate_rects[k].height / 2);

                  // Adjust rect to ensure it remains within image bounds
                  int rect_x = std::max(0, center.x - rect_width / 2);
                  int rect_y = std::max(0, center.y - rect_height / 2);
                  int rect_right = std::min(img_in.cols - 1, rect_x + rect_width);
                  int rect_bottom = std::min(img_in.rows - 1, rect_y + rect_height);

                  // Create rect 
                  cv::Rect bounding(rect_x, rect_y, rect_right - rect_x, rect_bottom - rect_y);

                  candidateSignCotours.push_back(Utils::getRectContours(bounding));
                  cv::rectangle(img_in, bounding, cv::Scalar(255, 0, 0), 2);
              }

              
          }

      }

     // ipa::imshow("img", img_in, true);
    }


    /**
     * @brief Function that handles all the Json stuff
     *
     */
    static void JsonHandler(bool show)
    {
        int ok = 0;
        int total_number = 96;
        for (int k = 1; k <= total_number; k++)
        {
            realSignContours.clear();
            candidateSignCotours.clear();

            std::string path = std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".jpg";
            std::string path_json = (std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".json");
            FILE* file = fopen(path_json.c_str(), "rb");
            // Definiamo un buffer di lettura
            char buffer[65536]; // 64KB

            // Creiamo uno stream di lettura dal file
            FileReadStream inputStream(file, buffer, sizeof(buffer));
            // printf("%s\n", path_json.c_str());

            // Definiamo un documento JSON
            Document document;

            // Parsiamo il documento JSON dall'input stream
            document.ParseStream(inputStream);

            // Chiudiamo il file
            fclose(file);
            // Estraiamo il campo desiderato (es. "campo")
            // Estraiamo i dati dal documento JSON
            cv::Mat img = cv::imread(path);

            cv::Mat img2draw = img.clone();

            for (auto& obj : document["objects"].GetArray())
            {
                if (strcmp(obj["label"].GetString(), noEntryLabelJson.c_str()) == 0)
                {
                    // printf("%s\n", obj["label"].GetString());

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

            if (show)
            {
                ipa::imshow("Real Sign", img2draw, true, 0.5f);
                ipa::imshow("Original image", img, true, 0.5f);
            }

            CoreFunctions::Preprocessing(img, show);
            float max_intersection = 0;
            
            for (int i = 0; i < realSignContours.size(); i++)
            {
                for (int j = 0; j < candidateSignCotours.size(); j++)
                {
                    float intersection = Utils::IntersectionOverUnion(Utils::verticesToRect(realSignContours[i]),
                        Utils::verticesToRect(candidateSignCotours[j]));
                    if (intersection > max_intersection)
                    {
                        max_intersection = intersection;
                    }
                }
            }
            
            if (max_intersection > 0.5)
            {
                std::string msg = std::string("Sign" + std::to_string(k) + ": ok");
                printf("%s\n", msg.c_str());
                ok++;
            }
            else
            {
                std::string msg = std::string("Sign" + std::to_string(k) + ": no");
                printf("%s\n", msg.c_str());
            }

            cv::destroyAllWindows();
        }
        printf("Number of ok: %d on %d\n", ok, total_number);
        int percentual = (ok / total_number);
        printf("Percentual %f", ((float)ok / total_number) * 100);
    }
};