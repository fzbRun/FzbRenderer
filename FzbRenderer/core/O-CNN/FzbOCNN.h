#pragma once

#ifndef FZB_O_CNN_H
#define FZB_O_CNN_H

#include<opencv/opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 

struct FzbOCNNConvolutionalLayer : public torch::nn::Module {

};

struct FzbOCNNModel : public torch::nn::Module{

	FzbOCNNModel() {
		
	}

};

struct FzbOCNN {

	template<typename T>
	void cudaArrayToTensor(T* array) {
		torch::Tensor input_tensor = torch::from_blob(
			cuda_int_array,                            // ����ָ��
			{ static_cast<int64_t>(num_elements) },       // ��״
			torch::dtype(torch::kInt32).device(torch::kCUDA) // �������ͺ��豸
		);
	}

};

#endif