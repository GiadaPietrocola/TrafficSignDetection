// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "utils.h"
#include "core_functions.h"

// include my project functions
#include "functions.h"

#include <iostream>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

using namespace rapidjson;

int main()
{

	// cv::Mat img = cv::imread("briaDataSet/00010201_ZM7zOyKxCfDUozcWKJj1WA.jpg");
	//	cv::Mat img = cv::imread(std::string(IMAGES_PATH) + "/Sign3.jpg");

	//	ipa::imshow("Original image", img, true, 0.5f);

	//	CoreFunctions::Preprocessing(img);

	for (int k = 1; k < 2; k++)
	{
		realSignContours.clear();
		candidateSgnCotours.clear();

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
			if (strcmp(obj["label"].GetString(), "regulatory--no-entry--g1") == 0)
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

		// printf("%s\n", path.c_str());

		ipa::imshow("Original image", img, true, 0.5f);
		CoreFunctions::Preprocessing(img);
		cv::destroyAllWindows();
	}
	return EXIT_SUCCESS;
}
