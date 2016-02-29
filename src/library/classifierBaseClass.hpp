#pragma once

#include "trainingExample.hpp"
#include<vector>
#include<tuple>
#include<string>

/**
Implementations of this class implement some classification method that uses training data.  It provides a uniform interface so it is simple to make a new classification method and add it to the test suite.
*/
class classifierBaseClass
{
public:
/**
This function returns the name of the classifier implementation.
*/
virtual std::string name() = 0;

/**
Implementations of this virtual function should take the given training data to get whatever data they need to do later classifications.
@param inputTrainingExamplesStartIterator: The start of the data to train with
@param inputTrainingExamplesEndIterator: The end of the data to train with
*/
virtual void train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator) = 0;

/**
This function makes the classifier implementation forget the information it got from training data (so that it can be trained again with a different set).
*/
virtual void reset() = 0;

/**
Implementations of this virtual function should classify the given images and compute the error rates of their classifications.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: Water/not water image classification error rates (error rate, false positive rate, false negative rate)
*/
virtual std::tuple<double, double, double> test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator) = 0;

/**
Implementations of this virtual function should segment the given images and calculate the resulting segmentation error rates.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: <the segmentations, per pixel error rate, per pixel false positive rate, per pixel false negative rate>
*/
virtual std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator) = 0;

virtual ~classifierBaseClass() {};
};
