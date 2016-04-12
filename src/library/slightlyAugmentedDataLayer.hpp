#include<caffe/caffe.hpp>
#include<caffe/layers/memory_data_layer.hpp> 
#include<vector>
#include<opencv2/opencv.hpp>

class slightlyAugmentedDataLayer : public caffe::MemoryDataLayer<float>
{
public:
slightlyAugmentedDataLayer(const caffe::LayerParameter& param);

//This function allows the data layer to be set with image inputs and outputs
void AddMatVector(const std::vector<cv::Mat>& inputMats, const std::vector<cv::Mat>& inputExpectedNetworkOutput);
};
