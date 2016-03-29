#pragma once

#include "classifierBaseClass.hpp"
#include "tiny_cnn.h"
#include "utilityFunctions.hpp"

//How to handle zero padding?
//Convert images to a format that tiny-cnn can use
//Train using images (conv or per pixel?) -> per pixel is basically conv with a single filter
//

class NeuralNetworkClassifier : public classifierBaseClass
{
public:

/**
This function returns the name of the classifier implementation.
*/
virtual std::string name();

/**
This function constructs a 3D pixel color histogram associated with water and not water pixels to approximate the PDF of water vs not water pixels given color.  It also computes the prior probability of any pixel being a water pixel.
@param inputTrainingExamplesStartIterator: The start of the data to train with
@param inputTrainingExamplesEndIterator: The end of the data to train with
*/
virtual void train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator);

/**
This function resets the classifier so it can be trained again.
*/
virtual void reset();

/**
This function classifies images as water/not water and then compares its classification with the labels associated with the test data.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: Water/not water image classification error rates (error rate, false positive rate, false negative rate)
*/
virtual std::tuple<double, double, double> test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator);

/**
This classifier supports per pixel segmentation, so it generates per pixel bitmasks indicating which pixels it thinks are water.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: <the segmentations, per pixel error rate, per pixel false positive rate, per pixel false negative rate>
*/
virtual std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator);

virtual ~NeuralNetworkClassifier() {}

/**
This function classifies the given example and returns what it thinks it is.
@param inputExample: The example to classify
@return: true if it thinks it is not water
*/
bool classify(const trainingExample &inputExample);

/**
This function segments a single image.
@param inputExample: The example to classify and segment
@return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate
*/
std::tuple<cv::Mat_<bool>, double, double, double> segment(const trainingExample &inputExample);

/**
This function classifies and segments a single image.
@param inputExample: The example to classify and segment
@return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate, classified as not water?
*/
std::tuple<cv::Mat_<bool>, double, double, double, bool> classifyAndSegment(const trainingExample &inputExample);


/**
This function segments a single image.
@param inputImage: The image to classify and segment
@return: segmentation (not water true)
*/
cv::Mat_<bool> segment(const cv::Mat &inputImage);


tiny_cnn::network<tiny_cnn::cross_entropy, tiny_cnn::gradient_descent> classifierNet;

double waterPixelPriorProbability;
double waterImagePriorProbability;
double fractionOfPixelsWaterInWaterImages;
double fractionOfPixelsWaterInNotWaterImages;

std::array<std::array<std::array<double, 256>, 256>, 256> waterPixelPDF;
std::array<std::array<std::array<double, 256>, 256>, 256> notWaterPixelPDF;
};


