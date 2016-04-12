#pragma once

#include<opencv2/opencv.hpp>
#include "tiny_cnn.h"
#include "trainingExample.hpp"
#include<caffe/caffe.hpp>

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

/** <- currently appears to take too much RAM, need to break up training data generation
This function decomposes training examples into small image patches.  This ramps up memory use considerably, so it would be better if there was another way to work with it using this library.
@param inputTrainingExamplesStartIterator: The start of the range of training examples to use
@param inputTrainingExamplesEndIterator: The end of the range of training examples to use
@param inputPatchSize: What size rectangular patch to make (must be odd) 
@return: <image patchs, expected results>
*/
std::array<std::vector<tiny_cnn::vec_t>, 2> decomposeTrainingExamplesAsPixelPatches(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator, int64_t inputPatchSize);

/** <- currently appears to take too much RAM, need to break up training data generation
This function decomposes training examples into small image patches.  This ramps up memory use considerably, so it would be better if there was another way to work with it using this library.
@param inputExample: A training example to decompose
@param inputPatchSize: What size rectangular patch to make (must be odd) 
@return: <image patchs, expected results>
*/
std::array<std::vector<tiny_cnn::vec_t>, 2> decomposeTrainingExampleAsPixelPatches(const trainingExample &inputExample, int64_t inputPatchSize);

/**
This function takes in a set of opencv images and reformats the data so that it can be handed to an appropriately sized blob via mutable_cpu_data.
@param inputImages: The images to convert
@param inputScalingFactor: How much to multiply the image by
@param inputTranslationFactor: How much to add to the image after scaling
@return: The data to use in the blob 
*/
std::vector<float> convertCVImagesToDataForBlob(const std::vector<cv::Mat> &inputImages, double inputScalingFactor = 1.0, double inputTranslationFactor = 0.0);


