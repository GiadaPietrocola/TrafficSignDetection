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

    static void Preprocessing(cv::Mat &img_in)
    {
        float img_area = img_in.rows * img_in.cols;
        cv::Mat blurred;
        cv::Mat edges;

        // Utils::RealCanny(img_in, blurred, edges, sigma);

        Utils::HistogramLabeq(img_in, img_in);

        cv::Mat img_blurred;
        Utils::CustomGaussianBlur(img_in, img_blurred, sigma);
        cv::imwrite(std::string(IMAGES_PATH) + "/gaussian.jpg", img_blurred);
        ipa::imshow("Gaussian", img_blurred, true, 0.5f);

        cv::Mat segmentedImage;
        Utils::RegionGrowingHSV(img_in, segmentedImage);

        //   cv::morphologyEx(segmentedImage, segmentedImage, cv::MORPH_CLOSE,
        //                          cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        ipa::imshow("Morph", segmentedImage, true, 0.5f);

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
        for (int i = 0; i < circular_objects.size(); i++)
        {
            cv::Rect boundingRect = cv::boundingRect(circular_objects[i]);
            candidateSignCotours.push_back(Utils::getRectContours(boundingRect));
            cv::rectangle(img_in, boundingRect, cv::Scalar(255, 0, 0), 2);
        }

        cv::drawContours(img_in, circular_objects, -1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

        ipa::imshow("image", img_in, true, 0.5f);

        std::vector<cv::Mat> roi;
        for (int k = 0; k < circular_objects.size(); k++)
        {
            cv::Rect boundingRect = cv::boundingRect(circular_objects[k]);
            roi.push_back(img_in(boundingRect).clone());

            ipa::imshow("ROI", roi[k], true);
        }

        // Applica il filtro di Canny per rilevare i bordi

        cv::Mat gray;
        cv::Mat tmp;
        std::vector<cv::Mat> roie;

        for (int k = 0; k < circular_objects.size(); k++)
        {
            Utils::RealCanny(roi[k], gray, tmp, sigma);
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

    /**
     * @brief Function that handles all the Json stuff
     *
     */
    static void JsonHandler()
    {
        for (int k = 1; k < 20; k++)
        {
            realSignContours.clear();
            candidateSignCotours.clear();

            std::string path = std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".jpg";
            std::string path_json = (std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".json");
            FILE *file = fopen(path_json.c_str(), "rb");
            // Definiamo un buffer di lettura
            char buffer[65536]; // 64KB

            // Creiamo uno stream di lettura dal file
            FileReadStream inputStream(file, buffer, sizeof(buffer));
            printf("%s\n", path_json.c_str());

            // Definiamo un documento JSON
            Document document;

            // Parsiamo il documento JSON dall'input stream
            document.ParseStream(inputStream);

            // Chiudiamo il file
            fclose(file);
            // Estraiamo il campo desiderato (es. "campo")
            // Estraiamo i dati dal documento JSON
            cv::Mat img = cv::imread(path);
            if (document.HasMember("width"))
            {
                std::cout << "Width: " << document["width"].GetInt() << std::endl;
            }
            cv::Mat img2draw = img.clone();

            for (auto &obj : document["objects"].GetArray())
            {
                if (strcmp(obj["label"].GetString(), noEntryLabelJson.c_str()) == 0)
                {
                    printf("%s\n", obj["label"].GetString());

                    std::vector<cv::Point> bbox;
                    rapidjson::Value bboxvalue = obj["bbox"].GetArray();
                    bbox.push_back(cv::Point(bboxvalue["xmin"].GetFloat(), bboxvalue["ymin"].GetFloat()));
                    bbox.push_back(cv::Point(bboxvalue["xmax"].GetFloat(), bboxvalue["ymin"].GetFloat()));
                    bbox.push_back(cv::Point(bboxvalue["xmax"].GetFloat(), bboxvalue["ymax"].GetFloat()));
                    bbox.push_back(cv::Point(bboxvalue["xmin"].GetFloat(), bboxvalue["ymax"].GetFloat()));
                    realSignContours.push_back(bbox);

                    cv::drawContours(img2draw, realSignContours, -1, cv::Scalar(0, 255, 0), 2);
                }
                ipa::imshow("Real Sign", img2draw, true, 0.5f);
            }

            ipa::imshow("Original image", img, true, 0.5f);
            // CoreFunctions::Preprocessing(img);
            cv::destroyAllWindows();
        }
    }
};