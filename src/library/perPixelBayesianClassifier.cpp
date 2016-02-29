#include "perPixelBayesianClassifier.hpp"

/**
This function returns the name of the classifier implementation.
*/
std::string perPixelBayesianClassifier::name()
{
return "perPixelBayesianClassifier";
}

/**
This function constructs a 3D pixel color histogram associated with water and not water pixels to approximate the PDF of water vs not water pixels given color.  It also computes the prior probability of any pixel being a water pixel.
@param inputTrainingExamplesStartIterator: The start of the data to train with
@param inputTrainingExamplesEndIterator: The end of the data to train with
*/
void perPixelBayesianClassifier::train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
int64_t pixelCount = 0;
int64_t waterPixelCount = 0;
int64_t notWaterPixelCount = 0;
int64_t waterImageCount = 0;
int64_t waterPixelCountInWaterImages = 0;
int64_t pixelCountInWaterImages = 0;
int64_t waterPixelCountInNotWaterImages = 0;
int64_t pixelCountInNotWaterImages = 0;

printf("\n\nTraining\n");
for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
printf("%s\n", iter->filename.c_str());
if(iter->isWaterImage)
{
waterImageCount++;
}
bool isNotWaterImage = !iter->isWaterImage;
const cv::Mat &sourceImage = iter->sourceImage;
const cv::Mat_<bool> &notWaterBitmap = iter->notWaterBitmap;

for(int64_t i=0; i<sourceImage.rows; i++)
{//Determine total brightness for averaging
for(int64_t a=0; a<sourceImage.cols; a++)
{
bool currentPixelIsNotWater = notWaterBitmap.at<bool>(i,a);
const cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) sourceImage.at<cv::Point3_<uchar> >(i,a);

if(currentPixelIsNotWater)
{
if(isNotWaterImage)
{
pixelCountInNotWaterImages++;
}
else
{
pixelCountInWaterImages++;
}

notWaterPixelPDF[pixelColor.x][pixelColor.y][pixelColor.z]++;
notWaterPixelCount++;
}
else
{
if(isNotWaterImage)
{
waterPixelCountInNotWaterImages++;
pixelCountInNotWaterImages++;
}
else
{
waterPixelCountInWaterImages++;
pixelCountInWaterImages++;
}

waterPixelCount++;
waterPixelPDF[pixelColor.x][pixelColor.y][pixelColor.z]++;
}

pixelCount++;
}
}
}

//Normalize PDFs
double waterNormalizationFactor = 1.0/waterPixelCount;
double notWaterNormalizationFactor = 1.0/notWaterPixelCount;

for(int i=0; i<waterPixelPDF.size(); i++)
{
for(int a=0; a<waterPixelPDF.size(); a++)
{
for(int b=0; b<waterPixelPDF.size(); b++)
{
waterPixelPDF[i][a][b] = waterPixelPDF[i][a][b]*waterNormalizationFactor;
notWaterPixelPDF[i][a][b] = notWaterPixelPDF[i][a][b]*notWaterNormalizationFactor;
}
}
}


fractionOfPixelsWaterInNotWaterImages = ((double) waterPixelCountInNotWaterImages) / pixelCountInNotWaterImages;

fractionOfPixelsWaterInWaterImages = ((double) waterPixelCountInWaterImages) / pixelCountInWaterImages;

waterPixelPriorProbability = ((double) waterPixelCount) / pixelCount;
waterImagePriorProbability = ((double) waterImageCount) / (inputTrainingExamplesEndIterator-inputTrainingExamplesStartIterator);
}

/**
This function resets the classifier so it can be trained again.
*/
void perPixelBayesianClassifier::reset()
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
std::tuple<double, double, double> perPixelBayesianClassifier::test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
int64_t numberOfTestExamples = inputTrainingExamplesEndIterator - inputTrainingExamplesStartIterator;
int64_t numberOfFalsePositives = 0;
int64_t numberOfFalseNegatives = 0;

printf("\n\nTesting\n");
for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
printf("%s\n", iter->filename.c_str());
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
std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> perPixelBayesianClassifier::calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
double errorRateSum = 0.0;
double falsePositiveRateSum = 0.0;
double falseNegativeRateSum = 0.0;
std::vector<cv::Mat_<bool>> segmentations;

printf("\n\nSegmenting\n");
for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
printf("%s\n", iter->filename.c_str());

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
bool perPixelBayesianClassifier::classify(const trainingExample &inputExample)
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
std::tuple<cv::Mat_<bool>, double, double, double> perPixelBayesianClassifier::segment(const trainingExample &inputExample)
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
std::tuple<cv::Mat_<bool>, double, double, double, bool> perPixelBayesianClassifier::classifyAndSegment(const trainingExample &inputExample)
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
cv::Mat_<bool> perPixelBayesianClassifier::segment(const cv::Mat &inputImage)
{
cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);

for(int64_t i=0; i<inputImage.rows; i++)
{
for(int64_t a=0; a<inputImage.cols; a++)
{
const cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) inputImage.at<cv::Point3_<uchar> >(i,a);
double waterPDF = waterPixelPDF[pixelColor.x][pixelColor.y][pixelColor.z]*waterPixelPriorProbability;
double notWaterPDF = notWaterPixelPDF[pixelColor.x][pixelColor.y][pixelColor.z]*(1.0-waterPixelPriorProbability);

bool isNotWater = waterPDF < notWaterPDF;
if(waterPDF == 0.0 && notWaterPDF == 0.0)
{
isNotWater = waterPixelPriorProbability < .5; //Take bigger prior if no pixel data available
}



segmentation.at<bool>(i,a) = isNotWater;
}
}

return segmentation;
}
