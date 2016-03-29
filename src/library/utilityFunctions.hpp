#pragma once

#include<opencv2/opencv.hpp>
#include "tiny_cnn.h"

/**
This function converts from a bool mat to a image that can be written out as a .png
@param inputBoolMat: The mat to convert
@return: The converted mat (white for true, black for false)
*/
cv::Mat_<cv::Point3_<uchar>> convertBoolMatToImage(const cv::Mat_<bool> &inputBoolMat);

/**
This function converts an OpenCV image to a format that a tiny_cnn neural network can work with ("row-wise" vector with inputs scaled to +- 1.0).
@param inputImage: The 3 channel image to convert
@param inputPaddingAmount: How many zero value pixels to add around the image
@return: The scaled vector to use with tiny_cnn
*/
tiny_cnn::vec_t convert3ChannelImageToTinyCNNVector(const cv::Mat_<cv::Point3_<uchar>> &inputImage, int64_t inputPaddingAmount = 0);

/**
This function converts an OpenCV image to a format that a tiny_cnn neural network can work with ("row-wise" vector with inputs scaled to +- 1.0).
@param inputImage: The bool image to convert
@param inputPaddingAmount: How many zero value pixels to add around the image
@return: The scaled vector to use with tiny_cnn
*/
tiny_cnn::vec_t convertBoolImageToTinyCNNVector(const cv::Mat_<bool> &inputImage, int64_t inputPaddingAmount = 0);
