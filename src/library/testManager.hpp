#pragma once

#include "trainingExample.hpp"
#include "classifierBaseClass.hpp"
#include<json.h>
#include<memory>
#include<fstream>
#include<cstdio>
#include<stdexcept>
#include<chrono>
#include<random>
#include<boost/filesystem.hpp>
#include "utilityFunctions.hpp"

/**
This class allows the training data specified by a JSON file to be loaded into memory and then passed to one 
*/
class testManager
{
public:
/**
This function intitializes the testManager with training data using the JSON index file at the given path.
@param inputTrainingDataIndexPath: The path to retrieve the JSON file from

@throws: This function can throw exceptions
*/
testManager(const std::string &inputTrainingDataIndexPath);

/**
This function adds a new classifier to the test manager so that it will be tested.
@param inputClassifier: The classifier to add to the test manager.  The test manager takes ownership of the classifiers, so it should be allocated on the heap and not deleted somewhere else.
*/
void addClassifier(classifierBaseClass &inputClassifier);

/**
This function shuffles the training data, presents part of it to the classifiers and then has the classifiers classify the test set.  It prints the performance of each of the classifiers to stdout and saves the segmentation results to folders in the given directory.
@param inputTestFraction: The fraction of the training data to use as test data
@param inputDirectoryToWriteSegmentationResultsTo: Where to write the results to

@throws: This function can throw exceptions
*/
void generateClassifierReports(double inputTestFraction, const std::string &inputDirectoryToWriteSegmentationResultsTo);

std::vector<trainingExample> trainingAndTestData;
std::deque<std::unique_ptr<classifierBaseClass>> classifiers; //Pointers to all of the classifiers to evaluate
};
