#include "NeuralNetworkClassifier.hpp"

using namespace tiny_cnn;

/**
This function returns the name of the classifier implementation.
*/
std::string NeuralNetworkClassifier::name()
{
return "NeuralNetworkClassifier";
}

/**
This function constructs a 3D pixel color histogram associated with water and not water pixels to approximate the PDF of water vs not water pixels given color.  It also computes the prior probability of any pixel being a water pixel.
@param inputTrainingExamplesStartIterator: The start of the data to train with
@param inputTrainingExamplesEndIterator: The end of the data to train with
*/
void NeuralNetworkClassifier::train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{


//Determine size of images (assumed to be the same)
std::array<int64_t, 2> imageDimensions{inputTrainingExamplesStartIterator->sourceImage.cols, inputTrainingExamplesStartIterator->sourceImage.rows};

std::vector<const trainingExample *> examplesReferences;
for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
examplesReferences.push_back(&(*iter));
}

//Create network architecture
int64_t imagePatchSize = 5;
classifierNet 
<< fully_connected_layer<activation::relu>(imagePatchSize*imagePatchSize*3, 75)
<< fully_connected_layer<activation::relu>(imagePatchSize*imagePatchSize*3, 75)
<< fully_connected_layer<activation::relu>(imagePatchSize*imagePatchSize*3, 75)
<< fully_connected_layer<activation::relu>(imagePatchSize*imagePatchSize*3, 75)
<< fully_connected_layer<activation::identity>(75, 1);

for(int i=0; i<1; i++)
{//Shuffle the image set so that it is presented in a different order each time
std::random_shuffle ( examplesReferences.begin(), examplesReferences.end() );
for(const trainingExample *exampleReference : examplesReferences)
{
//Should probably shuffle these too
std::array<std::vector<tiny_cnn::vec_t>, 2> imagePatchesAndPixelSegmentation = decomposeTrainingExampleAsPixelPatches(*exampleReference, imagePatchSize);

if(exampleReference == examplesReferences[0] && i == 0)
{//Reset weights on first run through
classifierNet.train(imagePatchesAndPixelSegmentation[0], imagePatchesAndPixelSegmentation[1], 100, 1, [](){}, [](){}, true);
}
else
{
classifierNet.train(imagePatchesAndPixelSegmentation[0], imagePatchesAndPixelSegmentation[1], 100, 1, [](){}, [](){}, false);
}
}
}

//Finished training, so save network in case we need it later
std::ofstream output("net.txt");
output << classifierNet;
}

/**
This function resets the classifier so it can be trained again.
*/
void NeuralNetworkClassifier::reset()
{

}

/**
This function classifies images as water/not water and then compares its classification with the labels associated with the test data.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: Water/not water image classification error rates (error rate, false positive rate, false negative rate)
*/
std::tuple<double, double, double> NeuralNetworkClassifier::test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{

return std::tuple<double, double, double>(-1.0,-1.0,-1.0);
}

/**
This classifier supports per pixel segmentation, so it generates per pixel bitmasks indicating which pixels it thinks are water.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: <the segmentations, per pixel error rate, per pixel false positive rate, per pixel false negative rate>
*/
std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> NeuralNetworkClassifier::calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
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
bool NeuralNetworkClassifier::classify(const trainingExample &inputExample)
{
return false;
}

/**
This function segments a single image.
@param inputExample: The example to classify and segment
@return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate
*/
std::tuple<cv::Mat_<bool>, double, double, double> NeuralNetworkClassifier::segment(const trainingExample &inputExample)
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
std::tuple<cv::Mat_<bool>, double, double, double, bool> NeuralNetworkClassifier::classifyAndSegment(const trainingExample &inputExample)
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
cv::Mat_<bool> NeuralNetworkClassifier::segment(const cv::Mat &inputImage)
{
cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);

for(int64_t i=0; i<inputImage.rows; i++)
{
for(int64_t a=0; a<inputImage.cols; a++)
{ 
//Create patch associated with pixel
int64_t imagePatchSize = 5;
tiny_cnn::vec_t currentPixelPatch(3*imagePatchSize*imagePatchSize,0.0);

for(int rowIndex = 0; rowIndex < imagePatchSize; rowIndex++) 
{ // Go over all rows
for(int columnIndex = 0; columnIndex < imagePatchSize; columnIndex++) 
{ // Go over all columns
if(((i+rowIndex-imagePatchSize/2) < 0) || ((i+rowIndex-imagePatchSize/2) > inputImage.rows) || ((a+columnIndex-imagePatchSize/2) < 0) || ((a+columnIndex-imagePatchSize/2) > inputImage.cols))
{
continue; //Moved outside image, so leave default
}
cv::Vec<uchar,3> pixel = inputImage.at<cv::Point3_<uchar>>(i+rowIndex-imagePatchSize/2, a+columnIndex-imagePatchSize/2);

//Make range for image -1.0 to 1.0
for(int pixelIndex = 0; pixelIndex < 3; pixelIndex++)
{
currentPixelPatch[imagePatchSize*imagePatchSize*pixelIndex+imagePatchSize*rowIndex+columnIndex] = (pixel[pixelIndex]*2.0)/255.0-1.0;
}

}
}

//Calculate if this is a water or not water image based on Neural net output
tiny_cnn::vec_t netOutput = classifierNet.predict(currentPixelPatch);


segmentation.at<bool>(i,a) = (netOutput[0] >= 0);
}
}

return segmentation;
}
