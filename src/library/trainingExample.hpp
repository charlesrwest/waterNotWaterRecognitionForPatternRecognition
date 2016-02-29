#pragma once

#include<opencv2/opencv.hpp>
#include<stdexcept>
#include<tuple>
#include<boost/filesystem.hpp>

/**
This class is used to represent a single training example in an easy to use manner.
*/
class trainingExample
{
public:
/**
This constructor loads the images from the given paths and uses them to initialize this training example.
@param inputSourceImagePath: The path to get the source image from
@param inputNotWaterBitmapPath: The path to get the bitmap indicating water/not water from
@param inputIsWater: True if this is considered a "water" image and false otherwise
*/
trainingExample(const std::string &inputSourceImagePath, const std::string &inputNotWaterBitmapPath, bool inputIsWater);

/**
This constructor initializes the image from in memory images.
@param inputSourceImage: The source image for this example
@param inputNotWaterBitmap: The 3 channel bitmap image
@param inputIsWater: True if this is considered a "water" image and false otherwise
*/
trainingExample(const cv::Mat &inputSourceImage, const cv::Mat &inputNotWaterBitmap, bool inputIsWater);

/**
This function computes the fraction of the time that a given classification of the pixels is wrong for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The error rate
*/
double calculateErrorFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest);

/**
This function computes the fraction of the time that a given classification of the pixels has a false positive for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The false positive rate
*/
double calculateFalsePositiveFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest);

/**
This function computes the fraction of the time that a given classification of the pixels has a false negative for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The false negative rate
*/
double calculateFalseNegativeFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest);

/**
This function computes the actual error counts
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: <error count, false positives count, false negatives count>
*/
std::tuple<int64_t, int64_t, int64_t> calculateErrors(const cv::Mat_<bool> &inputNotWaterBitmapToTest);

std::string filename;
cv::Mat sourceImage;
cv::Mat_<bool> notWaterBitmap; //True if not water, false if water with same positioning as source image
bool isWaterImage;
};
