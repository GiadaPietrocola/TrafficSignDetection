// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"
#include "core_functions.h"

// include my project functions
#include "functions.h"

int main()
{

	// cv::Mat img = cv::imread("briaDataSet/00010201_ZM7zOyKxCfDUozcWKJj1WA.jpg");
	//	cv::Mat img = cv::imread(std::string(IMAGES_PATH) + "/Sign3.jpg");

	//	ipa::imshow("Original image", img, true, 0.5f);

	//	CoreFunctions::Preprocessing(img);

	for (int k = 1; k < 30; k++)
	{
		std::string path = std::string(IMAGES_PATH) + "/Sign" + std::to_string(k) + ".jpg";
		printf("%s\n", path.c_str());
		cv::Mat img = cv::imread(path);
		ipa::imshow("Original image", img, true, 0.5f);
		CoreFunctions::Preprocessing(img);
		cv::destroyAllWindows();
	}
	return EXIT_SUCCESS;
}
