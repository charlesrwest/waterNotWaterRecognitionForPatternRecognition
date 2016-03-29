#include "utilityFunctions.hpp"

using namespace tiny_cnn;

/**
This function converts from a bool mat to a image that can be written out as a .png
@param inputBoolMat: The mat to convert
@return: The converted mat (white for true, black for false)
*/
cv::Mat_<cv::Point3_<uchar>> convertBoolMatToImage(const cv::Mat_<bool> &inputBoolMat)
{
cv::Mat_<cv::Point3_<uchar>> image(inputBoolMat.rows, inputBoolMat.cols);

for(int64_t i=0; i<inputBoolMat.rows; i++)
{
for(int64_t a=0; a<inputBoolMat.cols; a++)
{
if(inputBoolMat.at<bool>(i,a))
{
image.at<cv::Point3_<uchar> >(i,a) = cv::Point3_<uchar>(255,255,255);
}
else
{
image.at<cv::Point3_<uchar> >(i,a) = cv::Point3_<uchar>(0,0,0);
}
}
}

return image;
} 


/**
This function converts an OpenCV image to a format that a tiny_cnn neural network can work with ("row-wise" vector with inputs scaled to +- 1.0).
@param inputImage: The 3 channel image to convert
@param inputPaddingAmount: How many zero value pixels to add around the image
@return: The scaled vector to use with tiny_cnn
*/
tiny_cnn::vec_t convert3ChannelImageToTinyCNNVector(const cv::Mat_<cv::Point3_<uchar>> &inputImage, int64_t inputPaddingAmount)
{
double denominator = 255.0;
double min = -1.0;
double max = 1.0;
int64_t imageWidth = inputImage.cols;
int64_t imageHeight = inputImage.rows;

tiny_cnn::vec_t result(3*(imageHeight + 2*inputPaddingAmount)*(imageWidth + 2*inputPaddingAmount), min); //Set to min by default
for(int rowIndex = 0; rowIndex < imageHeight; rowIndex++) 
{ // Go over all rows
for(int columnIndex = 0; columnIndex < imageWidth; columnIndex++) 
{ // Go over all columns
for(int channelIndex = 0; channelIndex < 3; channelIndex++) 
{ // Go through all channels
cv::Vec<uchar,3> pixel = inputImage.at<cv::Point3_<uchar>>(rowIndex, columnIndex);

result[imageWidth*imageHeight*channelIndex + imageWidth*(rowIndex + inputPaddingAmount) + (columnIndex + inputPaddingAmount)] = pixel[channelIndex]/denominator*(max - min) + min;
}
}
}

return result;
}

/**
This function converts an OpenCV image to a format that a tiny_cnn neural network can work with ("row-wise" vector with inputs scaled to +- 1.0).
@param inputImage: The bool image to convert
@param inputPaddingAmount: How many zero value pixels to add around the image
@return: The scaled vector to use with tiny_cnn
*/
tiny_cnn::vec_t convertBoolImageToTinyCNNVector(const cv::Mat_<bool> &inputImage, int64_t inputPaddingAmount)
{
double denominator = 255.0;
double min = 0.0;
double max = 1.0;
int64_t imageWidth = inputImage.cols;
int64_t imageHeight = inputImage.rows;

tiny_cnn::vec_t result((imageHeight + 2*inputPaddingAmount)*(imageWidth + 2*inputPaddingAmount), min); //Set to min by default
for(int rowIndex = 0; rowIndex < imageHeight; rowIndex++) 
{ // Go over all rows
for(int columnIndex = 0; columnIndex < imageWidth; columnIndex++) 
{ // Go over all columns
result[imageWidth*(rowIndex + inputPaddingAmount) + (columnIndex + inputPaddingAmount)] = ((double) inputImage.at<bool>(rowIndex, columnIndex))/denominator*(max - min) + min;
}
}

return result;
}











