// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"

// include my project functions
#include "functions.h"

// include my project functions
#ifndef EXAMPLE_IMAGES_PATH
#define EXAMPLE_IMAGES_PATH "example_images"
#endif

struct CoreFunctions
{
    static void Preprocessing(cv::Mat &img_in)
    {
        float img_area = img_in.rows * img_in.cols;
        cv::Mat blurred;
        cv::Mat edges;

        Utils::RealCanny(img_in, blurred, edges, sigma);

        ipa::imshow("Edges", edges, true, 0.5f);

        // extract objects
        std::vector<std::vector<cv::Point>> objects;
        cv::findContours(edges, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

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

        // cv::Mat roi;
        // for (const auto &obj : circular_objects)
        // {
        // 	cv::Rect boundingRect = cv::boundingRect(obj);
        // 	roi = img(boundingRect).clone();
        // }
        // ipa::imshow("ROI", roi, true);
        // // Applica il filtro di Canny per rilevare i bordi

        // cv::Mat gray;
        // cv::Mat roie;
        // Utils::RealCanny(roi, gray, roie, sigma);

        // ipa::imshow("ROI", roie, true);

        // // extract objects
        // std::vector<std::vector<cv::Point>> robjects;
        // cv::findContours(roie, robjects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // // Calcola il bounding box per ciascun contorno e la differenza tra l'area del bounding box e l'area del contorno
        // for (size_t i = 0; i < robjects.size(); ++i)
        // {
        // 	// Calcola il bounding box del contorno
        // 	cv::Rect boundingRect = cv::boundingRect(robjects[i]);

        // 	if (boundingRect.width >= 20 && boundingRect.width <= 100)
        // 	{

        // 		// Calcola l'area del bounding box e l'area del contorno
        // 		double boundingArea = boundingRect.width * boundingRect.height;
        // 		double contourArea = cv::contourArea(robjects[i]);

        // 		// Calcola la differenza tra le aree
        // 		double areaDifference = boundingArea - contourArea;

        // 		std::cout << "Contour " << i << ": Bounding area = " << boundingArea
        // 				  << ", Contour area = " << contourArea
        // 				  << ", Area difference = " << areaDifference << std::endl;

        // 		if (areaDifference < 0.3 * boundingArea)
        // 		{
        // 			// Disegna il bounding box sulle immagine originale (solo per debugging)
        // 			cv::rectangle(roi, boundingRect, cv::Scalar(255), 1);
        // 		}
        // 	}
        // }

        // // Visualizza l'immagine con i bounding box disegnati
        // ipa::imshow("Bounding Boxes", roi, true);
    }
};