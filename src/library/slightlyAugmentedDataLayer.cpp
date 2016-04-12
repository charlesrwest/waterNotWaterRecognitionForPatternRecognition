#include "slightlyAugmentedDataLayer.hpp"

slightlyAugmentedDataLayer::slightlyAugmentedDataLayer(const caffe::LayerParameter& param) : MemoryDataLayer(param)
{
} 

void slightlyAugmentedDataLayer::AddMatVector(const std::vector<cv::Mat>& inputMats, const std::vector<cv::Mat>& inputExpectedNetworkOutput)
{
size_t num = inputMats.size();
MemoryDataLayer::added_data_.Reshape(num, channels_, height_, width_);
MemoryDataLayer::added_label_.Reshape(num, channels_, height_, width_);

// Apply data transformations (mirror, scale, crop...)
this->data_transformer_->Transform(inputMats, &added_data_);
this->data_transformer_->Transform(inputExpectedNetworkOutput, &added_label_);

//Officially add it
float* input_data = MemoryDataLayer::added_data_.mutable_cpu_data();
float* output_data = MemoryDataLayer::added_data_.mutable_cpu_data();

Reset(input_data, output_data, num);
has_new_data_ = true;
}
