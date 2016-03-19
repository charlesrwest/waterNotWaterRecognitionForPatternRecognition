#include "testManager.hpp"
#include "commandLineArgumentParser.hpp"
#include "perPixelBayesianClassifier.hpp"

int main(int argc, const char **argv)
{
//Read in arguments and make sure the right number are there
project::commandLineParser parser;
parser.parse(argv, argc);

if(!(parser.optionToAssociatedArguments.at(project::PROGRAM_STRING).size() >= 3))
{//Expecting program name, input directory, output directory
fprintf(stderr, "Error, expected: %s inputExamplesIndexPath outputDirectoryPath\n", argv[0]);
return 1;
}
const std::vector<std::string> &arguments = parser.optionToAssociatedArguments.at(project::PROGRAM_STRING);


std::string inputIndexPath = arguments[1]; 

std::string inputOutputDirectoryPath = arguments[2]; 

testManager testManagerInstance(inputIndexPath);

//Add one of these lines for each classifier class
testManagerInstance.addClassifier(*(new perPixelBayesianClassifier()));

//This makes output (number is fraction used for test)
testManagerInstance.generateClassifierReports(.33, inputOutputDirectoryPath);

return 0;
} 
