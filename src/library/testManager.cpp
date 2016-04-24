#include "testManager.hpp"

/**
This function intitializes the testManager with training data using the JSON index file at the given path.
@param inputTrainingDataIndexPath: The path to retrieve the JSON file from

@throws: This function can throw exceptions
*/
testManager::testManager(const std::string &inputTrainingDataIndexPath)
{
//Load the given JSON file
Json::Value index;
std::ifstream fileStream(inputTrainingDataIndexPath, std::ifstream::binary);
fileStream >> index;

if(!index.isMember("training_data"))
{
throw std::invalid_argument("Unable to read JSON index");
}

if(!index["training_data"].isArray())
{
throw std::invalid_argument("Unable to read JSON index");
}

Json::Value &trainingDataIndex = index["training_data"];
for(Json::Value::const_iterator iter = trainingDataIndex.begin(); iter != trainingDataIndex.end(); iter++)
{
const Json::Value &value = *iter;
if((!value.isMember("source_image_path")) || (!value.isMember("bitmask_path")) || (!value.isMember("is_water")))
{
throw std::invalid_argument("Unable to read JSON index");
}

if((!value["source_image_path"].isString()) || (!value["bitmask_path"].isString()) || (!value["is_water"].isBool()))
{
throw std::invalid_argument("Unable to read JSON index");
}

std::string imagePath = value["source_image_path"].asString();
std::string maskPath = value["bitmask_path"].asString();
bool isWater = value["is_water"].asBool();

trainingAndTestData.emplace_back(trainingExample(imagePath, maskPath, isWater));
}
}

/**
This function adds a new classifier to the test manager so that it will be tested.
@param inputClassifier: The classifier to add to the test manager.  The test manager takes ownership of the classifiers, so it should be allocated on the heap and not deleted somewhere else.
*/
void testManager::addClassifier(classifierBaseClass &inputClassifier)
{
classifiers.emplace_back(std::unique_ptr<classifierBaseClass>(&inputClassifier));
}

/**
This function shuffles the training data, presents part of it to the classifiers and then has the classifiers classify the test set.  It prints the performance of each of the classifiers to stdout and saves the segmentation results to folders in the given directory.
@param inputTestFraction: The fraction of the training data to use as test data
@param inputDirectoryToWriteSegmentationResultsTo: Where to write the results to

@throws: This function can throw exceptions
*/
void testManager::generateClassifierReports(double inputTestFraction, const std::string &inputDirectoryToWriteSegmentationResultsTo)
{
boost::filesystem::path outputDirectoryPath(inputDirectoryToWriteSegmentationResultsTo);

boost::filesystem::create_directory(outputDirectoryPath);

//Shuffle the training/test data (inefficient, but doesn't matter to much in this case)
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

shuffle(trainingAndTestData.begin(), trainingAndTestData.end(), std::default_random_engine(seed));

//shuffle(trainingAndTestData.begin(), trainingAndTestData.end(), std::default_random_engine(9809809853));//Shuffle, but the same way every time

//Disabled shuffling

int numberOfTestExamples = trainingAndTestData.size() * inputTestFraction;
int numberOfTrainingExamples = trainingAndTestData.size()-numberOfTestExamples;

std::vector<trainingExample>::const_iterator trainingDataEnd = trainingAndTestData.begin() + numberOfTrainingExamples;

for(const std::unique_ptr<classifierBaseClass> &classifierPointer : classifiers)
{
classifierBaseClass &classifier = *classifierPointer;

classifier.reset();

classifier.train(trainingAndTestData.begin(), trainingDataEnd);

double classificationErrorRate, classificationFalsePositiveRate, classificationFalseNegativeRate;


std::tie(classificationErrorRate, classificationFalsePositiveRate, classificationFalseNegativeRate) = classifier.test(trainingDataEnd, trainingAndTestData.end());

double pixelErrorRate, pixelFalsePositiveRate, pixelFalseNegativeRate;
std::vector<cv::Mat_<bool>> classifierSegmentations;

std::tie(classifierSegmentations, pixelErrorRate, pixelFalsePositiveRate, pixelFalseNegativeRate) =  classifier.calculateSegmentations(trainingDataEnd, trainingAndTestData.end());

printf("Classifier %s has classification error %lf", classifier.name().c_str(), classificationErrorRate);
if(classifierSegmentations.size() != 0)
{
printf(" and per pixel error %lf", pixelErrorRate);
}
printf("\n");

//Write out classifier segmentations

//Make directory in output directory to put the results for this classifier
boost::filesystem::path folderPath = outputDirectoryPath / classifier.name();

boost::filesystem::remove_all(folderPath);

boost::filesystem::create_directory(folderPath);



for(int i=0; i<classifierSegmentations.size(); i++)
{
cv::Mat_<cv::Point3_<uchar>> outputImage = convertBoolMatToImage(classifierSegmentations[i]);
boost::filesystem::path imagePath = folderPath / trainingAndTestData[numberOfTrainingExamples+i].filename;

cv::imwrite(imagePath.string(), outputImage);
}
}


}
