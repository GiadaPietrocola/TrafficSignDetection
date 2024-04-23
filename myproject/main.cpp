// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"
#include "core_functions.h"

// include my project functions
#include "functions.h"

// This is for vscode working
#ifndef IMAGES_PATH
#define IMAGES_PATH "example_images"
#endif

int main()
{

	// cv::Mat img = cv::imread("briaDataSet/00010201_ZM7zOyKxCfDUozcWKJj1WA.jpg");
	cv::Mat img = cv::imread(std::string(IMAGES_PATH) + "/Sign3.jpg");

	ipa::imshow("Original image", img, true, 0.5f);

	CoreFunctions::Preprocessing(img);

	return EXIT_SUCCESS;
}