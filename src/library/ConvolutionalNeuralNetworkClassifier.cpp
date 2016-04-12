#include "ConvolutionalNeuralNetworkClassifier.hpp"

using namespace tiny_cnn;
using caffe::Caffe;

/**
This function returns the name of the classifier implementation.
*/
std::string ConvolutionalNeuralNetworkClassifier::name()
{
return "ConvolutionalNeuralNetworkClassifier";
}

/**
This function constructs a 3D pixel color histogram associated with water and not water pixels to approximate the PDF of water vs not water pixels given color.  It also computes the prior probability of any pixel being a water pixel.
@param inputTrainingExamplesStartIterator: The start of the data to train with
@param inputTrainingExamplesEndIterator: The end of the data to train with
*/
void ConvolutionalNeuralNetworkClassifier::train(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{
std::vector<cv::Mat> trainingImages;
std::vector<cv::Mat> expectedNetworkOutput;
std::vector<int> fakeLabels;

for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
//3 Channel input images
trainingImages.emplace_back(inputTrainingExamplesStartIterator->sourceImage);
//inputTrainingExamplesStartIterator->sourceImage.convertTo(trainingImages.back(),CV_32FC3, 255.0);

expectedNetworkOutput.emplace_back(); //Convert to single channel float
inputTrainingExamplesStartIterator->notWaterBitmap.convertTo(expectedNetworkOutput.back(), CV_8UC1, 1.0);

fakeLabels.push_back(1);
}
std::vector<int> fakeLabels1 = fakeLabels;

//Create training network
//Set to run on CPU
Caffe::set_mode(Caffe::CPU);

//Read prototxt file to get network structure
caffe::SolverParameter solver_param;

caffe::ReadProtoFromTextFileOrDie("imageSegmentorNetSolver.prototxt", &solver_param);

//Create solver using loaded params
std::unique_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

//Load the data into the memory layer
caffe::MemoryDataLayer<float> &dataLayerImplementation = (*boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layers()[0]));

caffe::MemoryDataLayer<float> &dataLayerImplementation1 = (*boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layer_by_name("data1")));

dataLayerImplementation.AddMatVector(trainingImages, fakeLabels);

dataLayerImplementation1.AddMatVector(expectedNetworkOutput, fakeLabels1);

solver->Solve();

solver->Solve();

caffe::NetParameter serializedSolverNetwork;

solver->net()->ToProto(&serializedSolverNetwork);

//Remove layers marked for "train" mode, then add input
caffe::NetParameter filteredSerializedSolverNetwork;

//serializedSolverNetwork.mutable_state()->set_phase(caffe::TEST);

caffe::Net<float>::FilterNet(serializedSolverNetwork, &filteredSerializedSolverNetwork);

network.reset(new caffe::Net<float>("imageSegmentorNetTest.prototxt", caffe::TEST));
network->CopyTrainedLayersFrom(filteredSerializedSolverNetwork);

}

/**
This function resets the classifier so it can be trained again.
*/
void ConvolutionalNeuralNetworkClassifier::reset()
{

}

/**
This function classifies images as water/not water and then compares its classification with the labels associated with the test data.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: Water/not water image classification error rates (error rate, false positive rate, false negative rate)
*/
std::tuple<double, double, double> ConvolutionalNeuralNetworkClassifier::test(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const  std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
{

return std::tuple<double, double, double>(-1.0,-1.0,-1.0);
}

/**
This classifier supports per pixel segmentation, so it generates per pixel bitmasks indicating which pixels it thinks are water.
@param inputTrainingExamplesStartIterator: The start of the data to classify/calculate error rates for
@param inputTrainingExamplesEndIterator: The end of the data to classify/calculate error rates for
@return: <the segmentations, per pixel error rate, per pixel false positive rate, per pixel false negative rate>
*/
std::tuple<std::vector<cv::Mat_<bool>>, double, double, double> ConvolutionalNeuralNetworkClassifier::calculateSegmentations(const std::vector<trainingExample>::const_iterator &inputTrainingExamplesStartIterator, const std::vector<trainingExample>::const_iterator &inputTrainingExamplesEndIterator)
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
bool ConvolutionalNeuralNetworkClassifier::classify(const trainingExample &inputExample)
{
return false;
}

/**
This function segments a single image.
@param inputExample: The example to classify and segment
@return: segmentation (not water true), pixel error rate, pixel false positive rate, pixel false negative rate
*/
std::tuple<cv::Mat_<bool>, double, double, double> ConvolutionalNeuralNetworkClassifier::segment(const trainingExample &inputExample)
{
cv::Mat_<bool> segmentation;
double errorRate, falsePositiveRate, falseNegativeRate;
bool classifiedAsNotWater;

const cv::Mat &sourceImage = inputExample.sourceImage;
const cv::Mat_<bool> &notWaterBitmap = inputExample.notWaterBitmap;

int64_t falsePositiveCount = 0;
int64_t falseNegativeCount = 0;
int64_t numberOfPixelsInImage = sourceImage.rows*sourceImage.cols;

//cv::Mat_<bool> segmentation = segment(sourceImage);


//std::tie(segmentation, errorRate, falsePositiveRate, falseNegativeRate, classifiedAsNotWater) = classifyAndSegment(inputExample);

return std::tuple<cv::Mat_<bool>, double, double, double>(segmentation, errorRate, falsePositiveRate, falseNegativeRate);
}


/**
This function segments a single image.
@param inputImage: The image to classify and segment
@return: segmentation (not water true)
*/
cv::Mat_<bool> ConvolutionalNeuralNetworkClassifier::segment(const cv::Mat &inputImage)
{
cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);

/*
for(int64_t i=0; i<inputImage.rows; i++)
{
for(int64_t a=0; a<inputImage.cols; a++)
{ 
//Create patch associated with pixel
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

//printf("Net output: %lf\n", netOutput[0]);


segmentation.at<bool>(i,a) = (netOutput[0] >= .5);
}
}
*/

return segmentation;
}
