// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"
#include "core_functions.h"
#include <list>

// include my project functions
#include "functions.h"

// This is for vscode working
#ifndef EXAMPLE_IMAGES_PATH
#define EXAMPLE_IMAGES_PATH "example_images"
#endif

int main()
{

	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/giada4.jpeg");

	ipa::imshow("Original image", img, true, 0.5f);

	CoreFunctions::Preprocessing(img);

	return EXIT_SUCCESS;
}