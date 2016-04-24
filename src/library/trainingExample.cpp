#include "trainingExample.hpp"

/*
This constructor loads the images from the given paths and uses them to initialize this training example.
@param inputSourceImagePath: The path to get the source image from
@param inputNotWaterBitmapPath: The path to get the bitmap indicating water/not water from
@param inputIsWater: True if this is considered a "water" image and false otherwise
*/
trainingExample::trainingExample(const std::string &inputSourceImagePath, const std::string &inputNotWaterBitmapPath, bool inputIsWater) : isWaterImage(inputIsWater)
{
boost::filesystem::path imagePath(inputSourceImagePath);
filename = imagePath.filename().string();

//printf("Attempting to read: %s\n", imagePath.string().c_str());

sourceImage = cv::imread(inputSourceImagePath, CV_LOAD_IMAGE_COLOR);
if(sourceImage.data == nullptr)
{
throw std::invalid_argument("Unable to read image");
}

//printf("Attempting to read: %s\n", inputNotWaterBitmapPath.c_str());
cv::Mat notWaterBitmapRaw = cv::imread(inputNotWaterBitmapPath, CV_LOAD_IMAGE_COLOR);
if(notWaterBitmapRaw.data == nullptr)
{
throw std::invalid_argument("Unable to read image");
}

notWaterBitmap = cv::Mat_<bool>(notWaterBitmapRaw.rows, notWaterBitmapRaw.cols);
for(int64_t i=0; i<notWaterBitmapRaw.rows; i++)
{//Generate bool depending on water/not water
for(int64_t a=0; a<notWaterBitmapRaw.cols; a++)
{
cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) notWaterBitmapRaw.at<cv::Point3_<uchar> >(i,a);
notWaterBitmap.at<bool>(i,a) = (pixelColor == cv::Point3_<uchar>(0xff, 0xff, 0xff));
}
}
} 

/**
This constructor initializes the image from in memory images.
@param inputSourceImage: The source image for this example
@param inputNotWaterBitmap: The 3 channel bitmap image
@param inputIsWater: True if this is considered a "water" image and false otherwise
*/
trainingExample::trainingExample(const cv::Mat &inputSourceImage, const cv::Mat &inputNotWaterBitmap, bool inputIsWater) : sourceImage(inputSourceImage), notWaterBitmap(inputNotWaterBitmap), isWaterImage(inputIsWater)
{
}

/**
This function computes the fraction of the time that a given classification of the pixels is wrong for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The error rate
*/
double trainingExample::calculateErrorFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest)
{
int64_t errorCount = 0;
int64_t falsePositiveCount = 0;
int64_t falseNegativeCount = 0;

std::tie(errorCount, falsePositiveCount, falseNegativeCount) = calculateErrors(inputNotWaterBitmapToTest);

return ((double) errorCount) / (notWaterBitmap.rows*notWaterBitmap.cols);
}

/**
This function computes the fraction of the time that a given classification of the pixels has a false positive for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The false positive rate
*/
double trainingExample::calculateFalsePositiveFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest)
{
int64_t errorCount = 0;
int64_t falsePositiveCount = 0;
int64_t falseNegativeCount = 0;

std::tie(errorCount, falsePositiveCount, falseNegativeCount) = calculateErrors(inputNotWaterBitmapToTest);

return ((double) falsePositiveCount) / (notWaterBitmap.rows*notWaterBitmap.cols);
}

/**
This function computes the fraction of the time that a given classification of the pixels has a false negative for this example image.
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: The false negative rate
*/
double trainingExample::calculateFalseNegativeFraction(const cv::Mat_<bool> &inputNotWaterBitmapToTest)
{
int64_t errorCount = 0;
int64_t falsePositiveCount = 0;
int64_t falseNegativeCount = 0;

std::tie(errorCount, falsePositiveCount, falseNegativeCount) = calculateErrors(inputNotWaterBitmapToTest);

return ((double) falseNegativeCount) / (notWaterBitmap.rows*notWaterBitmap.cols);
}

/**
This function computes the actual error counts
@param inputNotWaterBitmapToTest: The pixel segmentation to evaluate.
@return: <error count, false positives count, false negatives count>
*/
std::tuple<int64_t, int64_t, int64_t> trainingExample::calculateErrors(const cv::Mat_<bool> &inputNotWaterBitmapToTest)
{
if(notWaterBitmap.rows != inputNotWaterBitmapToTest.rows || notWaterBitmap.cols != inputNotWaterBitmapToTest.cols)
{
throw std::invalid_argument("Invalid image");
}

int falsePositiveCount = 0;
int falseNegativeCount = 0;
for(int64_t i=0; i<notWaterBitmap.rows; i++)
{//Generate bool depending on water/not water
for(int64_t a=0; a<notWaterBitmap.cols; a++)
{
if(notWaterBitmap.at<bool>(i,a) == false && inputNotWaterBitmapToTest.at<bool>(i,a) == true)
{ //It is water and we think it is not
falseNegativeCount++;
}
else if(notWaterBitmap.at<bool>(i,a) == true && inputNotWaterBitmapToTest.at<bool>(i,a) == false)
{ //It is not water and we think it is
falsePositiveCount++;
}
}
}

return std::tuple<int64_t, int64_t, int64_t>(falsePositiveCount+falseNegativeCount, falsePositiveCount, falseNegativeCount);
}


