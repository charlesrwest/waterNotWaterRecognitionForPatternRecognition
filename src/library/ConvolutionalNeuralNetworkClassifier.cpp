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

std::vector<float> blobTrainingImages;
std::vector<float> blobExpectedOutput;
int64_t numberOfImages = 0;

{
std::vector<cv::Mat> trainingImages;
std::vector<cv::Mat> expectedNetworkOutput;

for(auto iter = inputTrainingExamplesStartIterator; iter != inputTrainingExamplesEndIterator; iter++)
{
trainingImages.emplace_back(inputTrainingExamplesStartIterator->sourceImage);

expectedNetworkOutput.emplace_back(); //Convert to single channel float
inputTrainingExamplesStartIterator->notWaterBitmap.convertTo(expectedNetworkOutput.back(), CV_8UC1, 1.0);
}

numberOfImages = trainingImages.size();

blobTrainingImages = convertCVImagesToDataForBlob(trainingImages, 2.0/255.0, -255.0/2.0); //Convert, scaling to -1 to 1 
blobExpectedOutput = convertCVImagesToDataForBlob(expectedNetworkOutput, 2.0, -.5); //Convert, scaling to -1 to 1
}

std::vector<float> dummyLabel(blobTrainingImages.size(), 1.0);

//Create training network
//Set to run on CPU
Caffe::set_mode(Caffe::GPU);

//Read prototxt file to get network structure
caffe::SolverParameter solver_param;

caffe::ReadProtoFromTextFileOrDie("imageSegmentorNetSolver.prototxt", &solver_param);

//Create solver using loaded params
std::unique_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

//Load the data into the memory layer
caffe::MemoryDataLayer<float> &dataLayerImplementation = (*boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layer_by_name("data")));

caffe::MemoryDataLayer<float> &dataLayerImplementation1 = (*boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layer_by_name("data1")));

printf("Blob image size: %ld\n", blobTrainingImages.size());

dataLayerImplementation.Reset(blobTrainingImages.data(), dummyLabel.data(), numberOfImages);

dataLayerImplementation1.Reset(blobExpectedOutput.data(), dummyLabel.data(), numberOfImages);

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
const cv::Mat &sourceImage = inputExample.sourceImage;
const cv::Mat_<bool> &notWaterBitmap = inputExample.notWaterBitmap;

int64_t falsePositiveCount = 0;
int64_t falseNegativeCount = 0;
int64_t numberOfPixelsInImage = sourceImage.rows*sourceImage.cols;

cv::Mat_<bool> segmentation = segment(sourceImage);

for(int64_t row=0; row<segmentation.rows; row++)
{
for(int64_t col=0; col<segmentation.cols; col++)
{
if(segmentation.at<bool>(row, col) && !notWaterBitmap.at<bool>(row, col))
{//False Positive
falsePositiveCount++;
}

if(!segmentation.at<bool>(row, col) && notWaterBitmap.at<bool>(row, col))
{//False negative
falseNegativeCount++;
}
}
}

double errorRate = (falsePositiveCount+falseNegativeCount)/((double) numberOfPixelsInImage);
double falsePositiveRate = falsePositiveCount/((double) numberOfPixelsInImage);
double falseNegativeRate = falseNegativeCount/((double) numberOfPixelsInImage);


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
//Load image into network
std::vector<float> convertedImage = convertCVImagesToDataForBlob(std::vector<cv::Mat>{inputImage}, 2.0/255.0, -255.0/2.0); //Convert, scaling to -1 to 1 

const std::vector< boost::shared_ptr< caffe::Blob<float> > > &inputNetBlobs = network->blobs();

caffe::Blob<float> &inputBlob = *inputNetBlobs[0];

float *array = inputBlob.mutable_cpu_data();
for(int i=0; i<convertedImage.size(); i++)
{
array[i] = convertedImage[i];
}
inputBlob.mutable_cpu_data(); //Make sure data gets syncronized
inputBlob.gpu_data();
inputBlob.mutable_gpu_data();

//Get output from network
const std::vector<caffe::Blob<float>*> &result = network->Forward();

//Convert it to a boolean image
cv::Mat_<bool> segmentation(inputImage.rows, inputImage.cols);

for(int64_t row=0; row<segmentation.rows; row++)
{
for(int64_t col=0; col<segmentation.cols; col++)
{
segmentation.at<bool>(row, col) = result[0]->cpu_data()[(segmentation.rows + row) * segmentation.cols + col] > 0;
}
}


return segmentation;
}
