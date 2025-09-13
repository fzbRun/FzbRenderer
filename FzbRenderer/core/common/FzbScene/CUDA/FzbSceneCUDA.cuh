#pragma once
#include <vector>
#include "../../FzbCommon.h"

#ifndef FZB_SCENE_CUDA_H
#define FZB_SCENE_CUDA_H

void addData_uint(uint32_t* data, uint32_t date, uint32_t dataNum);
void addPadding(std::vector<float>& sceneVertices, std::vector<float>& compressVertices, std::vector<uint32_t>& compressIndices, uint32_t& FzbVertexByteSize, FzbVertexFormat& vertexFormat);

#endif