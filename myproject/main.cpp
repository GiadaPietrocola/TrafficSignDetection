// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include <list>

// include my project functions
#include "functions.h"

// include my project functions
#include "functions.h"
#ifndef EXAMPLE_IMAGES_PATH
#define EXAMPLE_IMAGES_PATH "example_images"
#endif

int main()
{
	float sigma = 0.1;

	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/traffic_sign.jpg");
	ipa::imshow("Original image", img, true, 0.5f);

	cv::Mat blurred;
	cv::cvtColor(img, blurred, cv::COLOR_BGR2GRAY);

	int kernel_size = ucas::round(6 * sigma);
	if (kernel_size % 2 == 0)
		kernel_size++;

	cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), sigma);

	cv::Mat edges;
	cv::Canny(blurred, edges, 100 / 3, 100, 3, false);

	ipa::imshow("Edges", edges, true, 0.5f);

	// extract objects
	std::vector<std::vector<cv::Point>> objects;
	cv::findContours(edges, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	float min_circularity = 0.7;

	float min_width_perc = 0.02; // minimum width of light expressed as percentage of image width
	float max_width_perc = 0.5;	 // maximum width ...

	// specialized parameters
	int min_width = min_width_perc * img.cols + 0.5f;
	int max_width = max_width_perc * img.cols + 0.5f;

	// discard objects that are not in the [min_width, max_width]
	std::vector<std::vector<cv::Point>> candidate_objects;
	for (int k = 0; k < objects.size(); k++)
	{
		cv::Rect brect = cv::boundingRect(objects[k]);
		if (brect.width >= min_width && brect.width <= max_width)
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

	//	cv::drawContours(img, circular_objects, -1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

	ipa::imshow("image", img, true, 0.5f);

	cv::Mat roi;
	for (const auto &obj : circular_objects)
	{
		cv::Rect boundingRect = cv::boundingRect(obj);
		roi = img(boundingRect).clone();
	}
	ipa::imshow("ROI", roi, true);

	cv::Mat gray;
	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

	// Applica il filtro di Canny per rilevare i bordi
	cv::Mat roie;
	cv::Canny(gray, roie, 50, 150);

	ipa::imshow("ROI", roie, true);

	// extract objects
	std::vector<std::vector<cv::Point>> robjects;
	cv::findContours(roie, robjects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

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
				cv::rectangle(roi, boundingRect, cv::Scalar(255), 1);
			}
		}
	}

	// Visualizza l'immagine con i bounding box disegnati
	ipa::imshow("Bounding Boxes", roi, true);

	return EXIT_SUCCESS;
}