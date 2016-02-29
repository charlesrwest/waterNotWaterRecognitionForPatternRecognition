#pragma once

#include<opencv2/opencv.hpp>

/**
This function converts from a bool mat to a image that can be written out as a .png
@param inputBoolMat: The mat to convert
@return: The converted mat (white for true, black for false)
*/
cv::Mat_<cv::Point3_<uchar>> convertBoolMatToImage(const cv::Mat_<bool> &inputBoolMat);
