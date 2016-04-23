#include "SVMClassifier.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <sstream>
#define NTRAINING_SAMPLES       100                     // Number of training samples per class
#define FRAC_LINEAR_SEP         0.9f        // Fraction of samples which compose the linear separable part


using namespace cv;
using namespace std;


static CvSVM svm;
//using namespace CvSVM::train;

/**
  This function returns the name of the classifier implementation.
 */
std::string SVMClassifier::name()
{

	/*
	// Read in pixel data from training images
	std::ifstream pixelFile("./data/pixelLabel.dat");

	int lineCount = 0;

	if (pixelFile.is_open()) {
	while((! pixelFile.eof())&& (lineCount<10 )) {
	std::string strInput;
	pixelFile >> labels[lineCount];
	lineCount++;
	}

	pixelFile.close();
	} else {
	cout << "Unable to open file";
	return 0;
	}
	std::ifstream rgbFile("./data/RGB.dat");
	lineCount = 0;
	float trainingData[10][3];
	if(rgbFile.is_open()) {
	while((! rgbFile.eof()) && lineCount<10)) {
	rgbFile >> trainingData
	 */

	// Set up trainingpen() data
	// get the number of pixels there is in each image from above.
	// Need to parse Selense out put text to get the labels
	// Then read those into the label variable below. Go to a cpp tutorial.


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
	int64_t notWaterPixelCount = 0;
	int64_t waterImageCount = 0;
	int64_t waterPixelCountInWaterImages = 0;
	int64_t pixelCountInWaterImages = 0;
	int64_t waterPixelCountInNotWaterImages = 0;
	int64_t pixelCountInNotWaterImages = 0;
	//int64_t numberOfTestExamples = inputTrainingExamplesEndIterator - inputTrainingExamplesStartIterator;
	//int64_t numberOfTestExamples = 2;
	int width = 500, height = 333;
	//int totalPixels = ((int)numberOfTestExamples)* width*height;
	//int totalPixels = width*height;
	int totalPixels = 10000;
	cout << totalPixels << endl;
	float labels[totalPixels+1 ];
	float trainingData[totalPixels+1 ][3];
        cout << "about to start iterator" << endl;
	for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
	{
		if(iter->isWaterImage)
		{
			waterImageCount++;
		}
		bool isNotWaterImage = !iter->isWaterImage;
		const cv::Mat &sourceImage = iter->sourceImage;
		const cv::Mat_<bool> &notWaterBitmap = iter->notWaterBitmap;

		// ToDo: dynamically allocate the number of training pixels. for now we will just start with 20 pixels.
		for(int64_t i=0; i<sourceImage.rows; i++)
		{//Determine total brightness for averaging
			for(int64_t a=0; a<sourceImage.cols; a++)
			{
				// The variable below should be used to create labels matrix
				bool currentPixelIsNotWater  = notWaterBitmap.at<bool>(i,a);

				const cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) sourceImage.at<cv::Point3_<uchar> >(i,a);
				// This part was added. 20 should be changed later.
				if( (pixelCount < totalPixels+1) && ((currentPixelIsNotWater && (pixelCount%2==0)) || (!currentPixelIsNotWater && (pixelCount%2==1) )  ) ) {
			//	if ( pixelCount < totalPixels+1) {
					trainingData[pixelCount][0] = pixelColor.x;
					trainingData[pixelCount][1] = pixelColor.y;
					trainingData[pixelCount][2] = pixelColor.z;
				//	cout << trainingData[pixelCount] << " "; 
					labels[pixelCount] = notWaterBitmap.at<bool>(i,a);
				//	cout << labels[pixelCount] << " ";
				//	cout << pixelCount << " ";

					if (  !(labels[pixelCount])) {
						labels[pixelCount] = -1.0;
						//cout << labels[pixelCount];
						//cout << labels[pixelCount+1]<< endl;
						//    cout << currentPixelIsNotWater; 
						//    cout << waterImageCount<< endl; 
					} else {
						labels[pixelCount] = 1.0;
					}
                                          
				//	cout << labels[pixelCount]<< endl;
				        pixelCount++;
				}
			}	
		}

	}
	// just to make sure that the last test pixel belongs to another class
	cout << pixelCount << endl;
	// Once we have the training data, we put it in a basic struct
	Mat labelsMat(totalPixels, 1, CV_32FC1, labels);
	Mat trainingDataMat(totalPixels, 3, CV_32FC1, trainingData);
/*    const int WIDTH = 512, HEIGHT = 512;

    Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

    //--------------------- 1. Set up training data randomly ---------------------------------------
    Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32FC1);
    Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32FC1);

    RNG rng(100); // Random value generation class

    // Set up the linearly separable part of the training data
    int nLinearSamples = (int) (FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

    // Generate random points for the class 1
    Mat trainClass = trainData.rowRange(0, nLinearSamples);
    // The x coordinate of the points is in [0, 0.4)
    Mat c = trainClass.colRange(0, 1);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

    // Generate random points for the class 2
    trainClass = trainData.rowRange(2*NTRAINING_SAMPLES-nLinearSamples, 2*NTRAINING_SAMPLES);
    // The x coordinate of the points is in [0.6, 1]
    c = trainClass.colRange(0 , 1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
    //------------------ Set up the non-linearly separable part of the training data ---------------

    // Generate random points for the classes 1 and 2
    trainClass = trainData.rowRange(  nLinearSamples, 2*NTRAINING_SAMPLES-nLinearSamples);
    // The x coordinate of the points is in [0.4, 0.6)
    c = trainClass.colRange(0,1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

    //------------------------- Set up the labels for the classes ---------------------------------
    labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
    labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2

*/
	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	//params.gamma           = 2;
	//params.coef0           = 1;
	params.C           = .5;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, int(1e7), 1e-6);

	// Train the SVM
	cout << "Create SVM" << endl;
        cout << "Starting training process" << endl;
	svm.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
//        svm.train(trainData, labels, Mat(), Mat(), params);

	waterPixelPriorProbability = ((double) waterPixelCount) / pixelCount;
	waterImagePriorProbability = ((double) waterImageCount) / (inputTrainingExamplesEndIterator-inputTrainingExamplesStartIterator);
}

/**
  This function resets the classifier so it can be trained again.
 */
void SVMClassifier::reset()
{
	for(int i=0; i<waterPixelPDF.size(); i++)
	{
		for(int a=0; a<waterPixelPDF.size(); a++)
		{
			for(int b=0; b<waterPixelPDF.size(); b++)
			{
				waterPixelPDF[i][a][b] = 0.0;
				notWaterPixelPDF[i][a][b] = 0.0;
			}
		}
	}

	waterPixelPriorProbability = 0.0;
	waterImagePriorProbability = 0.0;
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

	//printf("\n\nTesting\n");
	for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
	{
		//printf("%s\n", iter->filename.c_str());
		bool classifiedAsNotWater = classify(*iter);

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
	cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);
	bool isNotWater;
	int totalPixels = (inputImage.rows)*(inputImage.cols);
	float inputData[totalPixels][3];
	int pixelCount = 0;
	//const cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) sourceImage.at<cv::Point3_<uchar> >(i,a);

	for(int64_t i=0; i<inputImage.rows; i++)
	{
		for(int64_t a=0; a<inputImage.cols; a++)
		{
			const cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) inputImage.at<cv::Point3_<uchar> >(i,a);
			inputData[pixelCount][0] = pixelColor.x;
			inputData[pixelCount][1] = pixelColor.y;
			inputData[pixelCount][2] = pixelColor.z;
			pixelCount++;

			Mat inputDataMat(1,3,CV_32FC1,inputData);	
			float response = svm.predict(inputDataMat);
			if (fabs(response + 1.0) < .5) 
			{
				isNotWater = 0;
			} else {
				isNotWater = 1;
			}
			//cout << response << endl;

			segmentation.at<bool>(i,a) = isNotWater;
		}
	}
       // create parameters for displaying output onto an image.
//	int height = 5;
//	int width = 2; 
//	Mat image = Mat::zeros(height, width, CV_8UC3);
//	Vec3b green(0,255,0), red(0,0,255);

	return segmentation;
}
