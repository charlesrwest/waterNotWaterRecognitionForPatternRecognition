#include "SVMClassifier.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define NTRAINING_SAMPLES       100                     // Number of training samples per class
#define FRAC_LINEAR_SEP         0.9f        // Fraction of samples which compose the linear separable part


using namespace cv;
using namespace std;


static CvSVM svm;
static int imgCount;
//using namespace CvSVM::train;

/**
  This function returns the name of the classifier implementation.
 */
std::string SVMClassifier::name()
{


	return "SVMClassifier";
}

/**
  This function constructs a 3D pixel color histogram associated with water and not water pixels to approximate the PDF of water vs not water pixels given color.  It also computes the prior probability of any pixel being a water pixel.
  @param inputTrainingExamplesStartIterator: The start of the data to train with
  @param inputTrainingExamplesEndIterator: The end of the data to train with
 */
void SVMClassifier::train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
	cout << "Start" << endl;
	int64_t pixelCount = 0;
	int64_t waterPixelCount = 0;
	//int64_t numberOfTestExamples = inputTrainingExamplesEndIterator - inputTrainingExamplesStartIterator;
	<<<<<<< HEAD
		int width = 500, height = 333;
	=======
		//int64_t numberOfTestExamples = 2;
		int width = 500, height = 333;
	//int totalPixels = ((int)numberOfTestExamples)* width*height;
	>>>>>>> 42ef2901b001366368b3db00cbe9ffa64027da31
		//int totalPixels = width*height;
		int totalPixels = 10000;
	cout << totalPixels << endl;
	float labels[totalPixels ];
	float trainingData[totalPixels][3];

	cout << "about to start iterator" << endl;
	for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
	{ //auto  iter = inputTrainingExamplesStartIterator; 

		printf("%s\n", iter->filename.c_str());

		const cv::Mat &sourceImage = iter->sourceImage;
		const cv::Mat_<bool> &notWaterBitmap = iter->notWaterBitmap;

		for(int64_t i=0; i<sourceImage.rows; i++)
		{//Determine total brightness for averaging
			for(int64_t a=0; a<sourceImage.cols; a++)
			{
				bool currentPixelIsNotWater  = notWaterBitmap.at<bool>(i,a);
				bool labels[pixelCount]      = notWaterBitmap.at<bool>(i,a);
				//const cv::Point3f &pixelColor =  sourceImage.at<cv::Point3f >(i,a);
				if( (pixelCount < totalPixels) && ((currentPixelIsNotWater && (pixelCount%2==0)) || (!currentPixelIsNotWater && (pixelCount%2==1) )  ) ) {
					//if ( pixelCount < totalPixels) {
					cout << " getting BGR values ";
					Vec3b intensity = sourceImage.at<Vec3b>(i,a);
					trainingData[pixelCount][0] = intensity[0];
					trainingData[pixelCount][1] = intensity[1];
					trainingData[pixelCount][2] = intensity[2];

					cout << trainingData[pixelCount][0] << " ";
					cout << trainingData[pixelCount][1] << " "; 			;
					cout << trainingData[pixelCount][2] << endl; 		

					if (  !(labels[pixelCount])) {
						labels[pixelCount] = -1.0;
					} else {
						labels[pixelCount] = 1.0;
					}

					pixelCount++;
				}	
				//}
			}


		}
		cout <<"pixel count " <<  pixelCount << endl;
		// Once we have the training data, we put it in a basic struct
		Mat labelsMat(totalPixels, 1, CV_32FC1, labels);
		Mat trainingDataMat(totalPixels, 3, CV_32FC1, trainingData);

		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		//params.gamma           = 2;
		//params.coef0           = 1;
		params.C           = 100;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, int(1e7), 1e-6);

		// Train the SVM
		cout << "Create SVM" << endl;
		cout << "Starting training process" << endl;
		svm.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	}
}

/**
  This function resets the classifier so it can be trained again.
 */
void SVMClassifier::reset()
{
}

/**
  This function classifies images as water/not water and then compares its classification with the labels associated with the test data.
  @param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
  @param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
  @return: Water/not water image classification error rates (error rate, false positive rate, false negative rate)
 */
std::tuple<double, double, double> SVMClassifier::test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
	int64_t numberOfTestExamples = inputTrainingExamplesEndIterator - inputTrainingExamplesStartIterator;
	int64_t numberOfFalsePositives = 0;
	int64_t numberOfFalseNegatives = 0;
	bool isNotWater;

	//printf("\n\nTesting\n");
	for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
	{
		const cv::Mat &sourceImage = iter->sourceImage;
		const cv::Mat_<bool> &notWaterBitmap = iter->notWaterBitmap;
		for(int64_t i=0; i<sourceImage.rows; i++)
		{
			for(int64_t a=0; a<sourceImage.cols; a++)
			{
				if (response < 0) 
				{
					isNotWater = 0;
					waterpxCount++;
				} else {
					isNotWater = 1;
					notWaterpxCount++;
				}
				bool classifiedAsNotWater = isNotWater;

				if(!classifiedAsNotWater)
				{
					if(!iter->isWaterImage)
					{
						numberOfFalseNegatives++;
					}
				}
				else
				{
					if(iter->isWaterImage)
					{
						numberOfFalsePositives++;
					}
				}
			}
		}
	}




	double averageFalsePositiveRate = ((double) numberOfFalsePositives) / numberOfTestExamples;
	double averageFalseNegativeRate = ((double) numberOfFalseNegatives) / numberOfTestExamples;
	double averageErrorRate = averageFalsePositiveRate+averageFalseNegativeRate;

	return std::tuple<double, double, double>(averageErrorRate, averageFalsePositiveRate, averageFalseNegativeRate);
}

/**
  This classifier supports per pixel segmentation, so it generates per pixel bitmasks indicating which pixels it thinks are water.
  @param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
  @param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
  @return: <the segmentations, per pixel error rate, per pixel false positive rate, per pixel false negative rate>
 */
std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> SVMClassifier::calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
	double errorRateSum = 0.0;
	double falsePositiveRateSum = 0.0;
	double falseNegativeRateSum = 0.0;
	std::vector<cv::Mat_<bool>> segmentations;

	//printf("\n\nSegmenting\n");
	for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
	{
		//printf("%s\n", iter->filename.c_str());

		double errorRate, falsePositiveRate, falseNegativeRate;
		cv::Mat_<bool> segmentation;
		std::tie(segmentation, errorRate, falsePositiveRate, falseNegativeRate) = segment(*iter);

		segmentations.push_back(segmentation);
		errorRateSum += errorRate;
		falsePositiveRateSum += falsePositiveRate;
		falseNegativeRateSum += falseNegativeRateSum;
	}



	double averageErrorRate = errorRateSum / segmentations.size();
	double averageFalsePositiveRate = falsePositiveRateSum / segmentations.size();
	double averageFalseNegativeRate = falseNegativeRateSum / segmentations.size();

	return std::tuple<std::vector<cv::Mat_<bool>>, double, double, double>(segmentations, averageErrorRate, averageFalsePositiveRate, averageFalseNegativeRate);
}

/**
  This function classifies the given example and returns what it thinks it is.
  @param inputExample: The example to classify
  @return: true if it thinks it is not water
 */
bool SVMClassifier::classify(const trainingExample &inputExample)
{
	cv::Mat_<bool> segmentation;
	double errorRate, falsePositiveRate, falseNegativeRate;
	bool classifiedAsNotWater;
	std::tie(segmentation, errorRate, falsePositiveRate, falseNegativeRate, classifiedAsNotWater) = classifyAndSegment(inputExample);

	return classifiedAsNotWater;
}

/**
  This function segments a single image.
  @param inputExample: The example to classify and segment
  @return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate
 */
std::tuple<cv::Mat_<bool>, double, double, double> SVMClassifier::segment(const trainingExample &inputExample)
{
	cv::Mat_<bool> segmentation;
	double errorRate, falsePositiveRate, falseNegativeRate;
	bool classifiedAsNotWater;
	std::tie(segmentation, errorRate, falsePositiveRate, falseNegativeRate, classifiedAsNotWater) = classifyAndSegment(inputExample);

	return std::tuple<cv::Mat_<bool>, double, double, double>(segmentation, errorRate, falsePositiveRate, falseNegativeRate);
}

/**
  This function classifies and segments a single image.
  @param inputExample: The example to classify and segment
  @return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate, classified as not water?
 */
std::tuple<cv::Mat_<bool>, double, double, double, bool> SVMClassifier::classifyAndSegment(const trainingExample &inputExample)
{
	const cv::Mat &sourceImage = inputExample.sourceImage;
	const cv::Mat_<bool> &notWaterBitmap = inputExample.notWaterBitmap;

	int64_t falsePositiveCount = 0;
	int64_t falseNegativeCount = 0;
	int64_t numberOfPixelsInImage = sourceImage.rows*sourceImage.cols;

	cv::Mat_<bool> segmentation = segment(sourceImage);

	//Count number of pixels considered water.  
	int64_t estimatedCountOfWaterPixelsInImage = 0;
	for(int64_t i=0; i<segmentation.rows; i++)
	{
		for(int64_t a=0; a<segmentation.cols; a++)
		{
			if(!segmentation.at<bool>(i,a))
			{
				estimatedCountOfWaterPixelsInImage++;
			}

			if(segmentation.at<bool>(i,a) == true)
			{
				if(notWaterBitmap.at<bool>(i,a) == false)
				{
					falsePositiveCount++;
				}
			}
			else
			{
				if(notWaterBitmap.at<bool>(i,a) == true)
				{
					falseNegativeCount++;
				}
			}

		}
	}

	double proportionOfPixelsWater = ((double) estimatedCountOfWaterPixelsInImage) / (numberOfPixelsInImage);

	//If the proportion of pixels classified water closer to that expected in a water image than it is in a not water image, classify it as a water image
	bool classifyAsWaterImage = false;
	if(fabs(fractionOfPixelsWaterInWaterImages - proportionOfPixelsWater) < fabs(fractionOfPixelsWaterInNotWaterImages - proportionOfPixelsWater))
	{
		classifyAsWaterImage = true;
	}

	double falsePositiveRate = ((double) falsePositiveCount) / numberOfPixelsInImage;
	double falseNegativeRate = ((double) falseNegativeCount) / numberOfPixelsInImage;
	double errorRate = falsePositiveRate + falseNegativeRate;


	return std::tuple<cv::Mat_<bool>, double, double, double, bool>(segmentation, errorRate, falsePositiveRate, falseNegativeRate, !classifyAsWaterImage);
}


/**
  This function segments a single image.
  @param inputImage: The image to classify and segment
  @return: segmentation (not water true)
 */
cv::Mat_<bool> SVMClassifier::segment(const cv::Mat &inputImage)
{
	<<<<<<< HEAD
		cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);
	bool isNotWater;

	bool isNotWater;
	int totalPixels = 333*500;
	float inputData[totalPixels][3];
	int pixelCount = 0;
	int waterpxCount = 0;
	int notWaterpxCount = 0;
	Mat input_mat(totalPixels,3,CV_32FC1);
	cout <<" total pixels "  << totalPixels;

	//printf("\n\nTesting\n");
	//for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)

	for(int64_t i=0; i<inputImage.rows; i++)
	{
		for(int64_t a=0; a<inputImage.cols; a++)
		{

			//printf("%s\n", iter->filename.c_str());
			Vec3b intensity = inputImage.at<Vec3b>(a,i);
			inputData[pixelCount][0]= intensity[0];
			inputData[pixelCount][1]= intensity[1];
			inputData[pixelCount][2]= intensity[2];
			cout << inputData[pixelCount][0] << " ";
			cout << inputData[pixelCount][1] << " "; 			;
			cout << inputData[pixelCount][2] << " "; 		
			Mat inputDataMat(1, 3, CV_32FC1, inputData);
			pixelCount++;
			cout << " pixelCount" << pixelCount;

			float response = svm.predict(inputDataMat);

			cout << " response : " << response << endl;
			segmentation.at<bool>(i,a) = response > 0;
		}
	}
	cout << " Image #" << imgCount;
	cout <<" total pixels "  << totalPixels;
	cout <<" water pixels "  << waterpxCount;
	cout <<" nonwater pixels "  << notWaterpxCount<< endl;
	imgCount++;

	return segmentation;
}
