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

/**
This function decomposes training examples into small image patches.  This ramps up memory use considerably, so it would be better if there was another way to work with it using this library.
@param inputTrainingExamplesStartIterator: The start of the range of training examples to use
@param inputTrainingExamplesEndIterator: The end of the range of training examples to use
@param inputPatchSize: What size rectangular patch to make (must be odd) 
@return: <image patchs, expected results>
*/
std::array<std::vector<tiny_cnn::vec_t>, 2> decomposeTrainingExamplesAsPixelPatches(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator, int64_t inputPatchSize)
{
if(inputPatchSize <= 0 || inputPatchSize % 2 != 1)
{
printf("%ld %ld\n", inputPatchSize, inputPatchSize % 2);
return std::array<std::vector<tiny_cnn::vec_t>, 2>();
}

std::array<std::vector<tiny_cnn::vec_t>, 2> results;
std::vector<tiny_cnn::vec_t> &imagePatch = results[0];
std::vector<tiny_cnn::vec_t> &expectedSegmentation = results[1];

for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
const cv::Mat &sourceImage = iter->sourceImage;
const cv::Mat_<bool> &notWaterBitmap = iter->notWaterBitmap;

//Zero fill if patch goes beyond edge
for(int64_t i=0; i<sourceImage.rows; i++)
{//Determine total brightness for averaging
for(int64_t a=0; a<sourceImage.cols; a++)
{ //Default to black for the image, true for notWater
imagePatch.emplace_back(tiny_cnn::vec_t(3*inputPatchSize,0.0));
tiny_cnn::vec_t &currentImagePatch = imagePatch.back();

expectedSegmentation.emplace_back(tiny_cnn::vec_t(1,notWaterBitmap.at<bool>(i,a))); //Set expected result for pixel


for(int rowIndex = 0; rowIndex < inputPatchSize; rowIndex++) 
{ // Go over all rows
for(int columnIndex = 0; columnIndex < inputPatchSize; columnIndex++) 
{ // Go over all columns
if(((i+rowIndex-inputPatchSize/2) < 0) || ((i+rowIndex-inputPatchSize/2) > sourceImage.rows) || ((a+columnIndex-inputPatchSize/2) < 0) || ((a+columnIndex-inputPatchSize/2) > sourceImage.cols))
{
continue; //Moved outside image, so leave default
}
cv::Vec<uchar,3> pixel = sourceImage.at<cv::Point3_<uchar>>(i+rowIndex-inputPatchSize/2, a+columnIndex-inputPatchSize/2);

//Make range for image -1.0 to 1.0
for(int pixelIndex = 0; pixelIndex < 3; pixelIndex++)
{
currentImagePatch[inputPatchSize*inputPatchSize*pixelIndex+inputPatchSize*rowIndex+columnIndex] = (pixel[pixelIndex]*2.0)/255.0-1.0;
}

}
}

}
}

printf("Processed image.  Result size: %ld\n", results.size());
}

return results;
}


/** <- currently appears to take too much RAM, need to break up training data generation
This function decomposes training examples into small image patches.  This ramps up memory use considerably, so it would be better if there was another way to work with it using this library.
@param inputExample: A training example to decompose
@param inputPatchSize: What size rectangular patch to make (must be odd) 
@return: <image patchs, expected results>
*/
std::array<std::vector<tiny_cnn::vec_t>, 2> decomposeTrainingExampleAsPixelPatches(const trainingExample &inputExample, int64_t inputPatchSize)
{
if(inputPatchSize <= 0 || inputPatchSize % 2 != 1)
{
printf("%ld %ld\n", inputPatchSize, inputPatchSize % 2);
return std::array<std::vector<tiny_cnn::vec_t>, 2>();
}

std::array<std::vector<tiny_cnn::vec_t>, 2> results;
std::vector<tiny_cnn::vec_t> &imagePatch = results[0];
std::vector<tiny_cnn::vec_t> &expectedSegmentation = results[1];

const cv::Mat &sourceImage = inputExample.sourceImage;
const cv::Mat_<bool> &notWaterBitmap = inputExample.notWaterBitmap;

//Zero fill if patch goes beyond edge
for(int64_t i=0; i<sourceImage.rows; i++)
{//Determine total brightness for averaging
for(int64_t a=0; a<sourceImage.cols; a++)
{ //Default to black for the image, true for notWater
imagePatch.emplace_back(tiny_cnn::vec_t(3*inputPatchSize*inputPatchSize,0.0));
tiny_cnn::vec_t &currentImagePatch = imagePatch.back();

expectedSegmentation.emplace_back(tiny_cnn::vec_t(1,notWaterBitmap.at<bool>(i,a))); //Set expected result for pixel

for(int rowIndex = 0; rowIndex < inputPatchSize; rowIndex++) 
{ // Go over all rows
for(int columnIndex = 0; columnIndex < inputPatchSize; columnIndex++) 
{ // Go over all columns
if(((i+rowIndex-inputPatchSize/2) < 0) || ((i+rowIndex-inputPatchSize/2) > sourceImage.rows) || ((a+columnIndex-inputPatchSize/2) < 0) || ((a+columnIndex-inputPatchSize/2) > sourceImage.cols))
{
continue; //Moved outside image, so leave default
}
cv::Vec<uchar,3> pixel = sourceImage.at<cv::Point3_<uchar>>(i+rowIndex-inputPatchSize/2, a+columnIndex-inputPatchSize/2);

//Make range for image -1.0 to 1.0
for(int pixelIndex = 0; pixelIndex < 3; pixelIndex++)
{
currentImagePatch[inputPatchSize*inputPatchSize*pixelIndex+inputPatchSize*rowIndex+columnIndex] = (pixel[pixelIndex]*2.0)/255.0-1.0;
}

}
}

}
}

printf("Processed image.  Result size: %ld\n", results[0].size());

return results;
}


/**
This function takes in a set of opencv images and reformats the data so that it can be handed to an appropriately sized blob via mutable_cpu_data.
@param inputImages: The images to convert
@return: The data to use in the blob 
*/
std::vector<float> convertCVImagesToDataForBlob(const std::vector<cv::Mat> &inputImages)
{
if(inputImages.size() == 0)
{
return std::vector<float>();
}

int64_t imageDepth = inputImages[0].depth();

std::vector<float> result(inputImages[0].rows*inputImages[0].cols*imageDepth);

for(int64_t imageIndex = 0; imageIndex < inputImages.size(); imageIndex++)
{
for(int64_t row=0; row<inputImages[imageIndex].rows; row++)
{
for(int64_t col=0; col<inputImages[imageIndex].cols; col++)
{
for(int64_t channel=0; channel < imageDepth; channel++)
{
result[((imageIndex * imageDepth + channel) * inputImages[imageIndex].rows + row) * inputImages[imageIndex].cols + col] = inputImages[imageIndex].at<float>(row, col, channel);
}
}
}
}

return result;
}





