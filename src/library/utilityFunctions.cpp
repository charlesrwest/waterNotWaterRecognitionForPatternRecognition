#include "utilityFunctions.hpp"

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
