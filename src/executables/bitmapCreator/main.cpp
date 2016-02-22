#include<boost/filesystem.hpp>
#include<cstdio>
#include<regex>
#include<opencv2/opencv.hpp>
#include "commandLineArgumentParser.hpp"

//Just finds pngs


/**
This function generates the actual bitmaps.
@param inputInputDirectoryPath: The directory to read images from
@param inputOutputDirectoryPath: The directory to write the bitmasks to
@return: true if successful
*/
bool generateMasks(const std::string &inputInputDirectoryPath, const std::string &inputOutputDirectoryPath);

/**
This function generates the mask for the given image and places it at the given path.
@param inputSourceImagePath: The path to read the image from
@param inputOutputImagePath: The path to write the mask to
@return: true if successful
*/
bool generateAndSaveMask(const std::string &inputSourceImagePath, const std::string &inputOutputImagePath);

/**
This function generates the mask for the given image.
@param inputSourceImagePath: The image to process
@return: The generated bitmask
*/
cv::Mat generateMask(const cv::Mat &inputSourceImage);

/**
This program takes a pair of directories.  Any .png files in the first directory are analysed and a matching bitmask .png image with black for portions of the image matching color 'ff0d0d' and white for everything else.  This can be used with other functionality to generate easy to use training data.
*/
int main(int argc, const char** argv )
{
//Test file system functionality and decide which library to use
boost::filesystem::path currentDirectory;
currentDirectory = boost::filesystem::current_path();

printf("Current path: %s\n", currentDirectory.c_str());

//Read in arguments and make sure the right number are there
project::commandLineParser parser;
parser.parse(argv, argc);
parser.printRetrievedOptionArgumentPairs();


if(!(parser.optionToAssociatedArguments.at(project::PROGRAM_STRING).size() >= 3))
{//Expecting program name, input directory, output directory
fprintf(stderr, "Error, expected: program inputDirectory outputDirectory\n");
return 1;
}
const std::vector<std::string> &arguments = parser.optionToAssociatedArguments.at(project::PROGRAM_STRING);


std::string inputDirectory = arguments[1];
std::string outputDirectory = arguments[2];

if(generateMasks(inputDirectory, outputDirectory) == false)
{
fprintf(stderr, "Unable to generate/save bitmasks\n");
return 1;
}




return 0;
} 


/**
This function generates the actual bitmaps.
@param inputInputDirectoryPath: The directory to read images from
@param inputOutputDirectoryPath: The directory to write the bitmasks to
@return: true if successful
*/
bool generateMasks(const std::string &inputInputDirectoryPath, const std::string &inputOutputDirectoryPath)
{
boost::filesystem::path inputDirectoryPath{inputInputDirectoryPath};
boost::filesystem::path outputDirectoryPath{inputOutputDirectoryPath}; 

if(!exists(outputDirectoryPath))
{
if(create_directory(outputDirectoryPath) == false)
{
fprintf(stderr, "Error, unable to create output directory\n");
return false;
}
}

for(auto it = boost::filesystem::directory_iterator(inputDirectoryPath); it != boost::filesystem::directory_iterator(); ++it)
{
const boost::filesystem::path &file = it->path();

auto hasRightFormat = [](const boost::filesystem::path& inputFile) 
{ 
return regex_search(inputFile.filename().string(), std::regex(".*\\.png"));
};

if(is_directory(file) || !hasRightFormat(file))
{ //Isn't a valid file
continue;
}

boost::filesystem::path outputFilePath = outputDirectoryPath / file.filename();

if(generateAndSaveMask(file.string(), outputFilePath.string()) == false)
{
return false;
}
}
}

/**
This function generates the mask for the given image and places it at the given path.
@param inputSourceImagePath: The path to read the image from
@param inputOutputImagePath: The path to write the mask to
@return: true if successful
*/
bool generateAndSaveMask(const std::string &inputSourceImagePath, const std::string &inputOutputImagePath)
{
cv::Mat sourceImage = cv::imread(inputSourceImagePath, CV_LOAD_IMAGE_COLOR);

if(sourceImage.data == nullptr)
{
return false;
}

cv::Mat outputImage = generateMask(sourceImage);

return imwrite(inputOutputImagePath, outputImage);
}

/**
This function generates the mask for the given image.
@param inputSourceImagePath: The image to process
@return: The generated bitmask
*/
cv::Mat generateMask(const cv::Mat &inputSourceImage)
{
cv::Mat outputImage(inputSourceImage.rows, inputSourceImage.cols, inputSourceImage.type());

cv::Point3_<uchar> colorToFind(0x0d, 0x0d, 0xff);

for(int64_t i=0; i<inputSourceImage.rows; i++)
{
for(int64_t a=0; a<inputSourceImage.cols; a++)
{
cv::Point3_<uchar> &pixelColor = (cv::Point3_<uchar>&) inputSourceImage.at<cv::Point3_<uchar> >(i,a);

if(pixelColor == colorToFind)
{
outputImage.at<cv::Point3_<uchar>>(i, a) = cv::Point3_<uchar>(0x0, 0x0, 0x0);
}
else
{
outputImage.at<cv::Point3_<uchar>>(i, a) = cv::Point3_<uchar>(0xff, 0xff, 0xff);
}

}
}

return outputImage;
}
