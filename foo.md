static class foo{
    - std::string path;
    - std::string path_json;
    - cv::Mat input_image;
    - std::vector<std::vector<cv::Point>> realSignContours;     // contours vector of real signs, taken from Json
    - std::vector<std::vector<cv::Point>> candidateSignCotours; // contours vector of candidate signs

}


- hough circle
- top hat 